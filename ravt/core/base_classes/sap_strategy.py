from typing import Callable, Tuple, Dict, Optional

import matplotlib.pyplot as plt

from ..configs import output_visualize_dir
from ..constants import BatchTDict, BatchNDict


class BaseSAPStrategy:
    def __init__(self, exp_tag: str):
        super().__init__()
        self.exp_tag = exp_tag
        self.process_time = []
        self.first_time = True
        self.max_time = 30

    def infer_sequence(
            self,
            input_fn: Callable[[], Optional[BatchNDict]],
            process_fn: Callable[[BatchTDict, Optional[Dict]], Tuple[BatchTDict, Dict]],
            output_fn: Callable[[BatchNDict], None],
            time_fn: Callable[[], float],
    ) -> None:
        if self.first_time:
            self.first_time = False

            def process_fn_record(*args, **kwargs):
                t1 = time_fn()
                res = process_fn(*args, **kwargs)
                t2 = time_fn()
                d = int(t1) - len(self.process_time)
                self.process_time.extend([0]*d)
                self.process_time.append(t2-t1)
                return res

            def output_fn_record(*args, **kwargs):
                return output_fn(*args, **kwargs)

            return self.infer_sequence_impl(
                input_fn=input_fn,
                process_fn=process_fn_record,
                output_fn=output_fn_record,
                time_fn=time_fn,
            )
        else:
            return self.infer_sequence_impl(
                input_fn=input_fn,
                process_fn=process_fn,
                output_fn=output_fn,
                time_fn=time_fn,
            )

    def infer_sequence_impl(
            self,
            input_fn: Callable[[], Optional[BatchNDict]],
            process_fn: Callable[[BatchTDict, Optional[Dict]], Tuple[BatchTDict, Dict]],
            output_fn: Callable[[BatchNDict], None],
            time_fn: Callable[[], float],
    ) -> None: raise NotImplementedError()

    def plot_process_time(self):
        plt.figure(dpi=300)
        plt.bar(list(range(len(self.process_time)))[:self.max_time], self.process_time[:self.max_time], width=0.5)
        plt.plot([-2, self.max_time+2], [1, 1], 'k--')
        plt.xlabel('Frame index')
        plt.ylabel('Process time / 33ms')
        plt.title(self.exp_tag)
        plt.savefig(str(output_visualize_dir.joinpath(f'{self.exp_tag.replace(" ", "_")}.png')))
        output_visualize_dir.joinpath(f'{self.exp_tag.replace(" ", "_")}.txt').write_text(
            '\n'.join([f'{t:.3f}' for t in self.process_time[:self.max_time]])
        )
