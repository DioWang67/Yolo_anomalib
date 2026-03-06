"""Thread-safe Overwrite FIFO Queue for high-throughput production pipelines.

Design Rationale
----------------
Standard ``queue.Queue`` blocks the producer when full, which is unacceptable
on a high-FPS production line because it stalls camera acquisition and causes
*all* downstream timing to drift.

``OverwriteQueue`` uses a bounded ``collections.deque`` so that when the buffer
is full the **oldest** item is silently discarded, keeping the producer
non-blocking.  Consumers use a standard blocking ``get()`` — they sleep until
a new item arrives or a timeout expires.

Concurrency Safety
------------------
All mutations are guarded by a single ``threading.Lock``.  A paired
``threading.Condition`` wakes blocked consumers when a new item is pushed.

Trade-off: We sacrifice O(1) ``get`` fairness (the lock is not a fair FIFO
lock on CPython) in exchange for a simple, auditable implementation.  For a
single-consumer topology this is perfectly safe.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OverwriteQueue(Generic[T]):
    """Bounded FIFO queue that drops the **oldest** item when full.

    Parameters
    ----------
    maxlen : int
        Maximum number of items the queue can hold.  Must be ≥ 1.
    name : str
        Human-readable label used in log messages (e.g. ``"inference_queue"``).

    Thread Safety
    -------------
    * ``put()``  — **never blocks**.  If the buffer is full the oldest item is
      evicted via ``deque``'s built-in ``maxlen`` behaviour.  A counter
      (``dropped_count``) tracks the total number of dropped items.
    * ``get()``  — **blocks** until an item is available or *timeout* expires.
      Raises ``queue.Empty`` on timeout (consistent with stdlib ``queue``).
    * ``qsize()`` / ``empty()`` — snapshot helpers (inherently racy but useful
      for monitoring / logging).

    Typical usage::

        q: OverwriteQueue[DetectionTask] = OverwriteQueue(maxlen=4)
        q.put(task)                 # producer — never blocks
        task = q.get(timeout=1.0)   # consumer — blocks up to 1 s
    """

    def __init__(self, maxlen: int, *, name: str = "OverwriteQueue") -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be ≥ 1, got {maxlen}")
        self._maxlen = maxlen
        self._name = name
        # deque(maxlen=N) automatically evicts the oldest item on append
        # when the buffer is full — O(1), thread-safe at the C level.
        # We still wrap mutations in a Lock to coordinate with Condition.
        self._buf: collections.deque[T] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

        # Observability counters (updated under lock for correctness).
        self.dropped_count: int = 0
        self._put_count: int = 0
        self._get_count: int = 0

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def put(self, item: T) -> None:
        """Enqueue *item*, evicting the oldest entry if the buffer is full.

        This method **never blocks**.

        Parameters
        ----------
        item : T
            The payload to enqueue.
        """
        with self._not_empty:
            if len(self._buf) >= self._maxlen:
                # deque.append will auto-evict, but we need to count it.
                self.dropped_count += 1
                logger.debug(
                    "[%s] Buffer full (%d/%d) — dropping oldest item "
                    "(total dropped: %d)",
                    self._name,
                    len(self._buf),
                    self._maxlen,
                    self.dropped_count,
                )

            self._buf.append(item)  # deque handles eviction if full
            self._put_count += 1
            self._not_empty.notify()  # wake one blocked consumer

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def get(self, timeout: float | None = None) -> T:
        """Remove and return the **oldest** item (FIFO order).

        Blocks until an item is available or *timeout* seconds elapse.

        Parameters
        ----------
        timeout : float | None
            Maximum seconds to wait.  ``None`` means wait forever.

        Returns
        -------
        T
            The dequeued item.

        Raises
        ------
        queue.Empty
            If *timeout* expires before an item becomes available.
        """
        with self._not_empty:
            # Spin-check inside Condition.wait to guard against spurious
            # wake-ups (documented CPython behaviour).
            while len(self._buf) == 0:
                if not self._not_empty.wait(timeout=timeout):
                    # wait() returned False → timed out
                    raise queue.Empty(
                        f"[{self._name}] get() timed out after {timeout}s"
                    )
            item = self._buf.popleft()
            self._get_count += 1
            return item

    # ------------------------------------------------------------------
    # Monitoring helpers
    # ------------------------------------------------------------------

    def qsize(self) -> int:
        """Return the approximate number of items in the queue."""
        with self._lock:
            return len(self._buf)

    def empty(self) -> bool:
        """Return ``True`` if the queue is currently empty."""
        with self._lock:
            return len(self._buf) == 0

    @property
    def maxlen(self) -> int:
        """The capacity of this queue."""
        return self._maxlen

    @property
    def put_count(self) -> int:
        """Total number of ``put()`` calls since creation."""
        return self._put_count

    @property
    def get_count(self) -> int:
        """Total number of successful ``get()`` calls since creation."""
        return self._get_count

    def __repr__(self) -> str:
        return (
            f"OverwriteQueue(name={self._name!r}, "
            f"size={self.qsize()}/{self._maxlen}, "
            f"puts={self._put_count}, gets={self._get_count}, "
            f"drops={self.dropped_count})"
        )
