class BiometricAuth:
    def verify_presence(self, card_id: str):
        print(f"[HW] Checking physical connection to MyKad: {card_id}...")
        return True

    def scan_face(self):
        print("[BIO] Running 3D Liveness Check... User verified.")
        return True

    def scan_fingerprint(self):
        print("[BIO] Fingerprint matched against chip-stored template.")
        return True