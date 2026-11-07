"""
Biometric Vector Encryption Module
Complies with biometric privacy standards (including UU PDP No. 27/2022)
Ensures face embeddings are encrypted at the application level before saving to Cloud Firestore.
"""

import base64
import json
from typing import List, Union
import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class BiometricEncryptionService:
    def __init__(self, key: str = None, version: str = None):
        self.version = version or settings.BIOMETRIC_KEY_VERSION
        raw_key = key or settings.BIOMETRIC_ENCRYPTION_KEY
        
        # Ensure key is valid URL-safe base64 32 bytes for Fernet
        try:
            self._cipher = Fernet(raw_key.encode() if isinstance(raw_key, str) else raw_key)
        except Exception as e:
            if settings.is_production:
                # Fail closed: a random fallback key would make every stored template undecryptable after restart.
                raise RuntimeError(
                    "BIOMETRIC_ENCRYPTION_KEY is invalid. Refusing to start in production."
                ) from e
            logger.warning(
                f"Configured BIOMETRIC_ENCRYPTION_KEY is invalid ({e}). Generating fallback development key."
            )
            self._cipher = Fernet(Fernet.generate_key())

    def encrypt_embedding(self, embedding: Union[np.ndarray, List[float]]) -> str:
        """
        Takes a 1D face embedding vector (numpy array or list of floats),
        serializes it to JSON, encrypts it, and returns a Base64-encoded ciphertext string.
        """
        if isinstance(embedding, np.ndarray):
            vector_list = embedding.astype(float).tolist()
        else:
            vector_list = [float(x) for x in embedding]

        payload = json.dumps({
            "vector": vector_list,
            "version": self.version,
            "dim": len(vector_list)
        })
        
        encrypted_bytes = self._cipher.encrypt(payload.encode("utf-8"))
        return encrypted_bytes.decode("utf-8")

    def decrypt_embedding(self, encrypted_embedding: str) -> np.ndarray:
        """
        Decrypts an encrypted embedding string and returns the original numpy float32 vector.
        Raises ValueError if tampering or invalid key is detected.
        """
        try:
            decrypted_bytes = self._cipher.decrypt(encrypted_embedding.encode("utf-8"))
            data = json.loads(decrypted_bytes.decode("utf-8"))
            vector = data["vector"]
            return np.array(vector, dtype=np.float32)
        except InvalidToken:
            logger.error("Failed to decrypt biometric embedding: Invalid token or key mismatch.")
            raise ValueError("Biometric decryption failed: Token is invalid or corrupted.")
        except Exception as e:
            logger.error(f"Unexpected error during embedding decryption: {e}")
            raise ValueError(f"Biometric decryption failed: {str(e)}")

# Singleton instance for application-wide use
biometric_crypto = BiometricEncryptionService()
