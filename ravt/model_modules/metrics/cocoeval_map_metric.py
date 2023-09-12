import platform
import contextlib
import functools
import io
import json
import tempfile
from typing import Dict, Optional

import torch
from pycocotools.coco import COCO

from ravt.protocols.structures import ConfigTypes, InternalConfigs
from ravt.utils.array_operations import xyxy2xywh
from ravt.utils.lightning_logger import ravt_logger
from ravt.protocols.structures import BatchDict, PredDict, MetricDict
from ravt.protocols.classes import BaseMetric


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
        )
        self.coco_factory = None
        self.seq_first_ids: Optional[Dict[int, int]] = None
        self.add_state('predict_list', default=[], dist_reduce_fx='cat')

    @functools.cached_property
    def coco(self) -> COCO:
        coco = self.coco_factory()
        if coco is None:
            raise ValueError(f'COCO factory returns None')
        else:
            return coco

    def phase_init_impl(self, phase: ConfigTypes, configs: InternalConfigs) -> InternalConfigs:
        if phase == 'evaluation':
            self.coco_factory = configs['evaluation']['coco_factory']
        return configs

    def update(
            self,
            batch: BatchDict,
            pred: PredDict,
            **kwargs
    ) -> None:
        # target is ignored
        original_size = batch['image']['original_size'][:, :1, None, :]
        current_size = torch.tensor(batch['image']['image'].shape[-2:], dtype=torch.float32, device=self.device)
        r = (original_size / current_size)[..., [1, 0, 1, 0]]

        pred_coordinates = xyxy2xywh(r*pred['bbox']['coordinate']).cpu().numpy()
        pred_labels = pred['bbox']['label'].cpu().numpy()
        pred_probabilities = pred['bbox']['probability'].cpu().numpy()
        image_ids = pred['bbox']['image_id'].cpu().numpy()

        for b in range(image_ids.shape[0]):
            for t in range(image_ids.shape[1]):
                i = image_ids[b, t].item()
                for c, l, p in zip(
                        pred_coordinates[b, t].tolist(),
                        pred_labels[b, t].tolist(),
                        pred_probabilities[b, t].tolist(),
                ):
                    if abs(p) > 1e-5:
                        self.predict_list.append({
                            'image_id': i,
                            'category_id': l,
                            'bbox': c,
                            'score': p,
                            'segmentation': [],
                        })

    def compute(self) -> MetricDict:
        # gt, outputs
        cocoGt = self.coco
        outputs = sorted(self.predict_list, key=lambda x: x['image_id'])

        if len(outputs) == 0:
            return {'mAP': torch.tensor(0.0, dtype=torch.float32, device=self.device)}

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
