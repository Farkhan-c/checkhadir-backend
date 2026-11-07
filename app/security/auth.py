"""
Authentication & Authorization Security Dependencies
Extracts and validates Firebase ID tokens and enforces Role-Based Access Control (RBAC).
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.firebase import (
    FirebaseServiceUnavailableError,
    verify_firebase_id_token_async,
    get_firestore_client,
)
from app.models.auth import TokenData
from app.models.user import UserRole
import logging

logger = logging.getLogger(__name__)
security_scheme = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
) -> TokenData:
    """
    Extracts and validates the Firebase ID token from the Authorization header.
    Returns TokenData with authoritative server-side user role and UID.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header or Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        decoded_token = await verify_firebase_id_token_async(token)
    except FirebaseServiceUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    uid = decoded_token.get("uid")
    email = decoded_token.get("email")
    name = decoded_token.get("name")
    
    # Check custom claims first, then check Firestore user profile for role & active status
    role = UserRole.EMPLOYEE
    db = get_firestore_client()
    
    if db is not None:
        try:
            user_doc = db.collection("users").document(uid).get()
            if user_doc.exists:
                user_data = user_doc.to_dict() or {}
                active_flag = user_data.get("isActive", user_data.get("is_active", True))
                if not active_flag:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="This employee account has been deactivated. Please contact your administrator."
                    )
                if user_data.get("role") == "admin":
                    role = UserRole.ADMIN
            else:
                # If custom claim has admin, fallback to it
                if decoded_token.get("role") == "admin":
                    role = UserRole.ADMIN
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error querying user profile for {uid}: {e}")
            if decoded_token.get("role") == "admin":
                role = UserRole.ADMIN
    else:
        # In dev or emulator mode, look at token claims
        if decoded_token.get("role") == "admin":
            role = UserRole.ADMIN

    return TokenData(
        uid=uid,
        email=email,
        name=name,
        role=role,
        claims=decoded_token
    )

async def require_admin(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """
    Enforces that the authenticated user holds the ADMIN role.
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Administrator privileges required."
        )
    return current_user

async def require_employee(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """
    Enforces that the caller is an authenticated employee or admin.
    """
    return current_user
