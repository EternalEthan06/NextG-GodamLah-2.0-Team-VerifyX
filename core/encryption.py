import os
from cryptography.fernet import Fernet

class EncryptionEngine:
    def __init__(self):
        # In production, this key is fetched from a Hardware Security Module (HSM)
        self.__key = os.getenv("KMS_KEY", Fernet.generate_key())
        self.__cipher = Fernet(self.__key)

    def encrypt_cell(self, data: str) -> str:
        """Encrypts a single column of data."""
        return self.__cipher.encrypt(data.encode()).decode()

    def decrypt_cell(self, ciphertext: str) -> str:
        """Decrypts data for authorized application use."""
        return self.__cipher.decrypt(ciphertext.encode()).decode()