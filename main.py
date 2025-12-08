from core.encryption import EncryptionEngine
from auth.biometrics import BiometricAuth
from storage.capsule import ShareCapsule
from core.audit import AuditManager # Assuming basic logger

class NationalIDApp:
    def __init__(self):
        self.engine = EncryptionEngine()
        self.bio = BiometricAuth()
        # Initialize Ali's Birth Cert in a secure capsule
        self.vault = ShareCapsule("Birth_Certificate", "BC-990101-14-1234", self.engine)

    def process_government_request(self):
        print("--- Transaction Started: JPN Agency ---")
        
        # Multilayer Verification Step
        if self.bio.verify_presence("801231-14-5566"):
            if self.bio.scan_face() and self.bio.scan_fingerprint():
                
                # Request Consent (Layer 04)
                self.vault.trigger_consent()
                
                # Decrypt only at the moment of use
                real_data = self.vault.unlock(self.engine)
                print(f"--- SUCCESS --- \nData retrieved: {real_data}")
            else:
                print("Security Breach: Biometric Mismatch")

if __name__ == "__main__":
    my_app = NationalIDApp()
    my_app.process_government_request()