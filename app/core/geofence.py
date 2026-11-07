"""
GPS Geofencing and Distance Calculation Module
Implements the Haversine formula to compute great-circle distance between coordinates.
Enforces office boundaries, geofence radius limits, and GPS accuracy thresholds.
"""

import math
from typing import Tuple, Optional
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

EARTH_RADIUS_METERS: float = 6371000.0

def calculate_haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculates the great-circle distance in meters between two GPS coordinates
    using the Haversine formula.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c

def validate_office_geofence(
    client_lat: float,
    client_lon: float,
    client_accuracy: float,
    office_lat: Optional[float] = None,
    office_lon: Optional[float] = None,
    radius_meters: Optional[float] = None,
    max_accuracy_meters: Optional[float] = None
) -> Tuple[bool, float, str]:
    """
    Validates whether client GPS coordinates fall within the authorized geofence radius.
    Returns (is_valid: bool, distance_meters: float, message: str).
    """
    target_lat = office_lat if office_lat is not None else settings.DEFAULT_OFFICE_LATITUDE
    target_lon = office_lon if office_lon is not None else settings.DEFAULT_OFFICE_LONGITUDE
    allowed_radius = radius_meters if radius_meters is not None else settings.DEFAULT_GEOFENCE_RADIUS_METERS
    allowed_accuracy = max_accuracy_meters if max_accuracy_meters is not None else settings.MAX_GPS_ACCURACY_METERS

    # 1. Validate GPS signal accuracy
    if client_accuracy > allowed_accuracy:
        return (
            False,
            0.0,
            f"Akurasi sinyal GPS perangkat terlalu rendah ({client_accuracy:.1f}m, maksimal {allowed_accuracy:.1f}m). "
            "Pastikan Anda berada di ruang terbuka atau aktifkan GPS berakurasi tinggi."
        )

    # 2. Compute Haversine distance
    distance = calculate_haversine_distance(client_lat, client_lon, target_lat, target_lon)

    # 3. Check within geofence boundary
    if distance > allowed_radius:
        return (
            False,
            round(distance, 1),
            f"Anda berada di luar radius presensi PAUD Laboratorium UM Kota Blitar "
            f"(jarak: {distance:.1f} meter, toleransi maksimal: {allowed_radius:.1f} meter)."
        )

    return (
        True,
        round(distance, 1),
        f"Lokasi valid: {distance:.1f} meter dari titik pusat sekolah."
    )
