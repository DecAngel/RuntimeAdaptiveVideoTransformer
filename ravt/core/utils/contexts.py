from typing import Callable


class CallOnExit:
    def __init__(self, *callables: Callable):
        self.callables = callables

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for c in self.callables:
            c()
