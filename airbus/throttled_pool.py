import time
import queue
from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import RUN
from multiprocessing.pool import IMapIterator
from multiprocessing.pool import mapstar

class ThrottledIMapIterator(IMapIterator):
    def _set(self, i, obj):
        with self._cond:
            if len(self._items) > 100: time.sleep(5)
            if self._index == i:
                self._items.append(obj)
                self._index += 1
                while self._index in self._unsorted:
                    obj = self._unsorted.pop(self._index)
                    self._items.append(obj)
                    self._index += 1
                self._cond.notify()
            else:
                self._unsorted[i] = obj

            if self._index == self._length:
                del self._cache[self._job]

class ThrottledPool(ThreadPool):
    def _setup_queues(self):
        self._inqueue = queue.Queue(100)
        self._outqueue = queue.Queue(100)
        self._quick_put = self._inqueue.put
        self._quick_get = self._outqueue.get

    def imap(self, func, iterable, chunksize=1):
        if self._state != RUN:
            raise ValueError("Pool not running")
        if chunksize == 1:
            result = ThrottledIMapIterator(self._cache)
            self._taskqueue.put(
                (
                    self._guarded_task_generation(result._job, func, iterable),
                    result._set_length
                ))
            return result
        else:
            assert chunksize > 1
            task_batches = Pool._get_tasks(func, iterable, chunksize)
            result = ThrottledIMapIterator(self._cache)
            self._taskqueue.put(
                (
                    self._guarded_task_generation(result._job,
                                                  mapstar,
                                                  task_batches),
                    result._set_length
                ))
            return (item for chunk in result for item in chunk)

