import platform
import contextlib
import io
import json
import tempfile
from typing import Optional, List

import torch
from pycocotools.coco import COCO

from ravt.core.constants import BatchTDict, MetricDict
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

    def __init__(self, future_time_constant: Optional[List[int]] = None):
        super().__init__(
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
            compute_on_cpu=True,
        )
        self.coco: Optional[COCO] = None
        self.future_time_constant = future_time_constant or [0]
        for i, t in enumerate(self.future_time_constant):
            self.add_state(f'image_id_list_{i}', default=[], dist_reduce_fx='cat')
            self.add_state(f'category_id_list_{i}', default=[], dist_reduce_fx='cat')
            self.add_state(f'bbox_list_{i}', default=[], dist_reduce_fx='cat')
            self.add_state(f'score_list_{i}', default=[], dist_reduce_fx='cat')

    def update(
            self,
            batch: BatchTDict,
            pred: BatchTDict,
            **kwargs
    ) -> None:
        # target is ignored
        original_size = batch['image']['original_size'][:, :1, None, :]     # (B, 1, 1, 2)
        current_size = torch.tensor(batch['image']['image'].shape[-2:], dtype=torch.float32, device=self.device)  # (2)
        r = (original_size / current_size)[..., [1, 0, 1, 0]]

        image_ids = (pred['image_id'][..., None] + pred['bbox']['clip_id']).cpu()
        category_ids = pred['bbox']['label'].cpu()
        bboxes = xyxy2xywh(r*pred['bbox']['coordinate']).cpu()
        scores = pred['bbox']['probability'].cpu()
        for i, (ii, c, b, s) in enumerate(zip(
                image_ids.unbind(1), category_ids.unbind(1), bboxes.unbind(1), scores.unbind(1)
        )):
            getattr(self, f'image_id_list_{i}').extend(ii.unbind(0))
            getattr(self, f'category_id_list_{i}').extend(c.unbind(0))
            getattr(self, f'bbox_list_{i}').extend(b.unbind(0))
            getattr(self, f'score_list_{i}').extend(s.unbind(0))

    def compute(self) -> MetricDict:
        # gt, outputs
        assert self.coco is not None
        cocoGt = self.coco

        # construct outputs
        res = {}
        for i, t in enumerate(self.future_time_constant):
            outputs = []
            for ii, c, b, p in zip(
                getattr(self, f'image_id_list_{i}'),
                getattr(self, f'category_id_list_{i}'),
                getattr(self, f'bbox_list_{i}'),
                getattr(self, f'score_list_{i}'),
            ):
                ii = ii.numpy().item()
                for _c, _b, _p in zip(c.numpy(), b.numpy(), p.numpy()):
                    if _p > 1e-5:
                        outputs.append({
                            'image_id': ii,
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

            m = torch.tensor(float(cocoEval.stats[0]), dtype=torch.float32, device=self.device)
            res[f'mAP_{t}'] = m
            if i == 0:
                res['mAP'] = m
        return res
