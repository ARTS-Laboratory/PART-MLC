""" Derivation of Queue that allows iteration."""
from multiprocessing import Queue


class IterQueue(Queue):
    def __init__(self, timeout=None, stop_val=None, stop_count=1):
        super(IterQueue, self).__init__()
        self.timeout = timeout
        self.stop_val = stop_val
        self.num_stops = 0
        self.stop_count = stop_count

    def __iter__(self):
        self.item = super().get(timeout=self.timeout)
        return self

    def __next__(self):
        if self.num_stops < self.stop_count:
            val = self.item
            self.item = super().get(timeout=self.timeout)
            if val == self.stop_val:
                self.num_stops += 1
            return val
        else:
            raise StopIteration

