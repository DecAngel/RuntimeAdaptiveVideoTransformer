import platform
import contextlib
import functools
import io
import itertools
import json
import tempfile
from typing import Dict, Optional, List, TypedDict

import typeguard
import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics import Metric
from pycocotools.coco import COCO
from jaxtyping import Float, Int

from ravt.common.array_operations import xyxy2xywh
from ravt.common.lightning_logger import ravt_logger


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


class COCOEvalMAPMetric(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    class OutputTypedDict(TypedDict):
        mAP: float

    def __init__(self):
        super().__init__(
            compute_on_cpu=True,
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
        )
        self.set_fn: Optional[functools.partial] = None
        self.coco: Optional[COCO] = None
        self.seq_first_ids: Optional[Dict[int, int]] = None
        # self.predict_list: List = []
        self.add_state('predict_list', default=[], dist_reduce_fx=None)

    def set_coco(self, coco: COCO):
        self.coco = coco

        first_img_id = None
        first_seq_id = None
        seq_lens = []
        for i, img in enumerate(self.coco.loadImgs(self.coco.getImgIds())):
            if i == 0:
                first_img_id = img['id']
                first_seq_id = img['sid']
                seq_lens.append(0)
            elif img['sid'] - first_seq_id == len(seq_lens):
                seq_lens.append(0)
            seq_lens[-1] += 1
        self.seq_first_ids = {
            i + first_seq_id: f
            for i, f in enumerate(itertools.accumulate([first_img_id] + seq_lens[:-1]))
        }

    def setup(self, trainer: pl.Trainer, stage: Optional[str] = None):
        def set_coco(t: pl.Trainer, s: Optional[str] = None):
            if s in ['fit', 'validate']:
                try:
                    self.set_coco(t.val_dataloaders.dataset.coco)
                except TypeError or IndexError or AttributeError as e:
                    raise RuntimeError(
                        f'Eval dataset does not contain `coco` attribute, cannot use COCOEvalMetric!'
                    ) from e
            elif stage == 'test':
                try:
                    self.set_coco(t.test_dataloaders.dataset.coco)
                except TypeError or IndexError or AttributeError as e:
                    raise RuntimeError(
                        f'Test dataset does not contain `coco` attribute, cannot use COCOEvalMetric!'
                    ) from e
        self.set_fn = functools.partial(set_coco, trainer, stage)

    @typeguard.typechecked()
    def update(
            self,
            image_ids: Int[torch.Tensor, 'batch_size'],
            resize_ratios: Float[torch.Tensor, 'batch_size ratios=2'],
            pred_coordinates: Float[torch.Tensor, 'batch_size max_objs coords=4'],
            pred_probabilities: Float[torch.Tensor, 'batch_size max_objs'],
            pred_labels: Int[torch.Tensor, 'batch_size max_objs'],
            **kwargs
    ) -> None:
        # target is ignored
        if self.coco is None:
            self.set_fn()

        for coordinates, probabilities, labels, image_id, r in zip(
                pred_coordinates.detach().cpu().numpy(),
                pred_probabilities.detach().cpu().numpy(),
                pred_labels.detach().cpu().numpy(),
                image_ids.detach().cpu().numpy(),
                resize_ratios.detach().cpu().numpy(),
        ):
            r = np.stack([r[1], r[0]]*2, axis=0)
            coordinates = xyxy2xywh(coordinates*r)
            for l, c, p in zip(labels.tolist(), coordinates.tolist(), probabilities.tolist()):
                if abs(p) > 1e-5:
                    self.predict_list.append({
                        'image_id': image_id.item(),
                        'category_id': l,
                        'bbox': c,
                        'score': p,
                        'segmentation': [],
                    })

    @typeguard.typechecked()
    def compute(self) -> OutputTypedDict:
        # gt, outputs
        cocoGt = self.coco
        outputs = sorted(self.predict_list, key=lambda x: x['image_id'])

        if len(outputs) == 0:
            return {'mAP': 0.0}

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

        return {'mAP': float(cocoEval.stats[0])}
