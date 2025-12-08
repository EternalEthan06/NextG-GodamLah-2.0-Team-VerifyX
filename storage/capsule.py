class ShareCapsule:
    def __init__(self, document_name, data, engine):
        self.doc_name = document_name
        self.__encrypted_data = engine.encrypt_cell(data)
        self.consent_granted = False

    def trigger_consent(self):
        self.consent_granted = True
        print(f"[CONSENT] User authorized sharing of {self.doc_name}.")

    def unlock(self, engine):
        if not self.consent_granted:
            raise PermissionError("Access Denied: No User Consent.")
        return engine.decrypt_cell(self.__encrypted_data)