"""
Per-user sliding-window rate limiter for expensive biometric endpoints.

Face detection/matching is CPU-heavy, so unthrottled calls are both a DoS vector and an
online brute-force vector against face verification. Limits are keyed by authenticated UID.

NOTE: State is in-process memory. With several Cloud Run instances each instance enforces its
own window, so the effective limit is (limit x instances). Deploy with a low max-instances
for the 30-100 employee scale, or move this to Redis/Firestore if you scale out.
"""

import time
import threading
from collections import defaultdict, deque
from typing import Deque, Dict, Callable
from fastapi import Depends, HTTPException, status

from app.models.auth import TokenData
from app.security.auth import get_current_user


class SlidingWindowLimiter:
    def __init__(self, window_seconds: int = 60):
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int) -> int:
        """Records a hit. Returns 0 if allowed, else the seconds until the caller may retry."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= limit:
                return max(1, int(self.window - (now - q[0])) + 1)
            q.append(now)
            return 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


_limiter = SlidingWindowLimiter(window_seconds=60)


def rate_limit(bucket: str, limit_getter: Callable[[], int]) -> Callable:
    """FastAPI dependency factory: `Depends(rate_limit("face", lambda: settings.X))`."""

    async def _dependency(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        limit = limit_getter()
        retry_after = _limiter.check(f"{bucket}:{current_user.uid}", limit)
        if retry_after:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Terlalu banyak permintaan. Coba lagi dalam {retry_after} detik.",
                headers={"Retry-After": str(retry_after)},
            )
        return current_user

    return _dependency


def reset_rate_limits() -> None:
    """Test helper."""
    _limiter.reset()
