import platform
import contextlib
import io
import json
import tempfile
from typing import Dict, Optional

import torch
from pycocotools.coco import COCO

from ravt.core.constants import PhaseTypes, AllConfigs, BatchDict, PredDict, MetricDict
from ravt.core.utils.array_operations import xyxy2xywh
from ravt.core.utils.lightning_logger import ravt_logger
from ravt.core.base_classes import BaseMetric


coco_eval_version = ""
try:
    if platform.system() == 'Linux':
        from .yolox_cocoeval import COCOeval_opt as COCOeval
        coco_eval_version = 'Using YOLOX COCOeval.'
    else:
        from pycocotools.cocoeval import COCOeval
        coco_eval_version = 'Using pycocotools COCOeval.'
except ImportError:
    from pycocotools.cocoeval import COCOeval
    coco_eval_version = 'Using pycocotools COCOeval.'
ravt_logger.info(coco_eval_version)


class COCOEvalMAPMetric(BaseMetric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self):
        super().__init__(
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
            compute_on_cpu=True,
        )
        self.coco: Optional[COCO] = None
        self.available: bool = False
        self.add_state('image_id_list', default=[], dist_reduce_fx='cat')
        self.add_state('category_id_list', default=[], dist_reduce_fx='cat')
        self.add_state('bbox_list', default=[], dist_reduce_fx='cat')
        self.add_state('score_list', default=[], dist_reduce_fx='cat')

    def phase_init_impl(self, phase: PhaseTypes, configs: AllConfigs) -> AllConfigs:
        if phase == 'evaluation':
            if 'stage' in configs['internal']:
                stage = configs['internal']['stage']
                if stage in ['fit', 'evaluation']:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.coco = COCO(str(configs['extra']['eval_coco_file']))
                elif stage == 'test':
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.coco = COCO(str(configs['extra']['test_coco_file']))
                else:
                    raise ValueError(f'No coco file for stage {stage}')
                self.available = True
        return configs

    def update(
            self,
            batch: BatchDict,
            pred: PredDict,
            **kwargs
    ) -> None:
        # target is ignored
        assert self.available
        original_size = batch['image']['original_size'][:, :1, None, :]
        current_size = torch.tensor(batch['image']['image'].shape[-2:], dtype=torch.float32, device=self.device)
        r = (original_size / current_size)[..., [1, 0, 1, 0]]

        self.image_id_list.extend(pred['bbox']['image_id'].cpu().flatten(0, 1).unbind(0))
        self.category_id_list.extend(pred['bbox']['label'].cpu().flatten(0, 1).unbind(0))
        self.bbox_list.extend(xyxy2xywh(r*pred['bbox']['coordinate']).cpu().flatten(0, 1).unbind(0))
        self.score_list.extend(pred['bbox']['probability'].cpu().flatten(0, 1).unbind(0))

    def compute(self) -> MetricDict:
        # gt, outputs
        assert self.available
        cocoGt = self.coco

        # construct outputs
        outputs = []
        for i, c, b, p in zip(
            self.image_id_list,
            self.category_id_list,
            self.bbox_list,
            self.score_list,
        ):
            i = i.numpy().item()
            for _c, _b, _p in zip(c.numpy(), b.numpy(), p.numpy()):
                if _p > 1e-5:
                    outputs.append({
                        'image_id': i,
                        'category_id': _c.item(),
                        'bbox': _b.tolist(),
                        'score': _p.item(),
                        'segmentation': [],
                    })

        if len(outputs) == 0:
            return {'mAP': torch.tensor(0.0, dtype=torch.float32, device=self.device)}
        else:
            outputs = sorted(outputs, key=lambda x: x['image_id'])

        res_str = io.StringIO()
        with contextlib.redirect_stdout(res_str):
            # pred
            _, tmp = tempfile.mkstemp()
            json.dump(outputs, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)

            # eval
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()

            cocoEval.summarize()

        return {'mAP': torch.tensor(float(cocoEval.stats[0]), dtype=torch.float32, device=self.device)}
