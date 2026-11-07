"""
Canonical Server Timezone Utilities — Waktu Indonesia Barat (WIB / Asia/Jakarta, UTC+07:00).
Ensures consistent WIB server timestamps across authentication, attendance, employee join dates,
reports, leaves, and application logging regardless of the host container OS timezone.
"""

from datetime import datetime, timedelta, timezone
import os
import time

# Ensure POSIX-aware libraries see Asia/Jakarta by default
os.environ.setdefault("TZ", "Asia/Jakarta")
if hasattr(time, "tzset"):
    try:
        time.tzset()
    except Exception:
        pass

# Fixed UTC+7 offset (Indonesia Western Standard Time — Asia/Jakarta observes no DST)
WIB = timezone(timedelta(hours=7), name="WIB")


def now_wib() -> datetime:
    """Returns timezone-aware current server datetime in WIB (UTC+07:00)."""
    return datetime.now(WIB)


def today_wib_str() -> str:
    """Returns current calendar date in WIB formatted as YYYY-MM-DD."""
    return now_wib().strftime("%Y-%m-%d")


def to_wib(dt: datetime) -> datetime:
    """Converts any naive (assumed UTC) or timezone-aware datetime to WIB (UTC+07:00)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(WIB)
