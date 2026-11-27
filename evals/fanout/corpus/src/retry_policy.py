"""Retry with exponential backoff, bounded jitter and an overall deadline.

:class:`RetryPolicy` is an immutable description of a backoff schedule.
Attempts are numbered from 1.  After failed attempt ``n`` the nominal pause
is ``base_delay * multiplier ** (n - 1)``; jitter then scales the pause by a
factor drawn uniformly from ``[1 - jitter, 1 + jitter]``, and the result is
capped so that no pause is ever longer than ``max_delay``.

:class:`Retrier` applies a policy to a callable.  It stops retrying when the
callable succeeds, when ``max_attempts`` attempts have been made, or when the
next pause would end after the policy deadline, measured from the start of
the first attempt.  The sleep function, the clock and the random source are
injectable so that the behaviour can be tested without waiting.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Type


class RetryError(Exception):
    """Raised when a call is abandoned; wraps the last failure."""

    def __init__(self, attempts: int, last_exception: BaseException, reason: str) -> None:
        super().__init__(f"gave up after {attempts} attempt(s): {reason}")
        self.attempts = attempts
        self.last_exception = last_exception
        self.reason = reason


@dataclass(frozen=True)
class RetryPolicy:
    """Backoff schedule parameters.

    Attributes:
        base_delay: nominal pause after the first failed attempt, in seconds.
        multiplier: growth factor applied per further failed attempt.
        max_delay: upper bound for any single pause, jitter included.
        jitter: relative jitter in ``[0, 1)``; 0 disables jitter.
        max_attempts: total number of attempts, the first one included.
        deadline: optional budget in seconds for the whole call.
    """

    base_delay: float = 0.1
    multiplier: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.0
    max_attempts: int = 5
    deadline: Optional[float] = None

    def __post_init__(self) -> None:
        if self.base_delay < 0:
            raise ValueError("base_delay must not be negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")
        if self.max_delay < 0:
            raise ValueError("max_delay must not be negative")
        if not 0 <= self.jitter < 1:
            raise ValueError("jitter must lie in [0, 1)")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.deadline is not None and self.deadline <= 0:
            raise ValueError("deadline must be positive")

    def nominal_delay(self, attempt: int) -> float:
        """Uncapped, jitter-free pause after failed attempt ``attempt``."""
        if attempt < 1:
            raise ValueError("attempt numbers start at 1")
        if self.base_delay == 0:
            return 0.0
        try:
            return self.base_delay * self.multiplier ** (attempt - 1)
        except OverflowError:
            return float("inf")

    def delay_bounds(self, attempt: int) -> Tuple[float, float]:
        """Smallest and largest pause possible after failed attempt ``attempt``."""
        nominal = self.nominal_delay(attempt)
        low = min(self.max_delay, nominal * (1 - self.jitter))
        high = min(self.max_delay, nominal * (1 + self.jitter))
        return low, high

    def delay(self, attempt: int, rng: Callable[[], float] = random.random) -> float:
        """Pause to observe after failed attempt ``attempt``.

        ``rng`` must return floats in ``[0, 1)``.  The result always lies
        within :meth:`delay_bounds` and never exceeds ``max_delay``.
        """
        nominal = self.nominal_delay(attempt)
        if self.jitter == 0:
            return min(self.max_delay, nominal)
        factor = 1 - self.jitter + 2 * self.jitter * rng()
        return min(self.max_delay, nominal) * factor

    def schedule(self, rng: Callable[[], float] = random.random) -> List[float]:
        """Pauses between consecutive attempts if every attempt fails.

        The list has ``max_attempts - 1`` entries and ignores the deadline.
        """
        return [self.delay(attempt, rng) for attempt in range(1, self.max_attempts)]

    def worst_case_total(self) -> float:
        """Upper bound on the time spent sleeping when every attempt fails."""
        total = sum(self.delay_bounds(attempt)[1] for attempt in range(1, self.max_attempts))
        if self.deadline is not None:
            total = min(total, self.deadline)
        return total


class Retrier:
    """Runs callables under a :class:`RetryPolicy`."""

    def __init__(
        self,
        policy: RetryPolicy,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: Callable[[], float] = random.random,
        retry_on: Tuple[Type[BaseException], ...] = (Exception,),
        on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    ) -> None:
        """Create a retrier.

        Args:
            policy: the backoff schedule.
            sleep: called with the pause in seconds before each retry.
            clock: returns the current time in seconds.
            rng: random source for jitter, returning floats in ``[0, 1)``.
            retry_on: exception types that trigger a retry; anything else
                propagates immediately.
            on_retry: optional observer called with the failed attempt
                number, the exception and the pause about to be slept.
        """
        self._policy = policy
        self._sleep = sleep
        self._clock = clock
        self._rng = rng
        self._retry_on = retry_on
        self._on_retry = on_retry
        self._last_attempts = 0

    @property
    def policy(self) -> RetryPolicy:
        """The policy this retrier applies."""
        return self._policy

    @property
    def last_attempts(self) -> int:
        """Number of attempts made by the most recent :meth:`call`."""
        return self._last_attempts

    def should_retry(self, attempt: int, elapsed: float, pause: float) -> Optional[str]:
        """Decide whether to retry after failed attempt ``attempt``.

        Returns ``None`` to go ahead, or the reason for giving up.
        """
        if attempt >= self._policy.max_attempts:
            return "attempts exhausted"
        deadline = self._policy.deadline
        if deadline is not None and elapsed + pause > deadline:
            return "deadline exceeded"
        return None

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call ``func`` until it succeeds or the policy gives up.

        Raises:
            RetryError: the policy was exhausted; ``__cause__`` and
                ``last_exception`` hold the final failure.
        """
        started = self._clock()
        attempt = 0
        while True:
            attempt += 1
            self._last_attempts = attempt
            try:
                return func(*args, **kwargs)
            except self._retry_on as exc:
                pause = self._policy.delay(attempt, self._rng)
                elapsed = self._clock() - started
                reason = self.should_retry(attempt, elapsed, pause)
                if reason is not None:
                    raise RetryError(attempt, exc, reason) from exc
                if self._on_retry is not None:
                    self._on_retry(attempt, exc, pause)
                self._sleep(pause)

    def wrap(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Return a function that runs ``func`` through :meth:`call`."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(func, *args, **kwargs)

        wrapper.__name__ = getattr(func, "__name__", "wrapped")
        wrapper.__doc__ = func.__doc__
        return wrapper
