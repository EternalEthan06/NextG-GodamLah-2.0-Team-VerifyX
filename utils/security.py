import hashlib
import json
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

class MerkleTree:
    """
    Layer 1 Security: Merkle Tree Implementation
    Anchors document hashes to a root hash to ensure integrity and sequence.
    """
    def __init__(self):
        self.leaves = []
        self.root = None

    def add_leaf(self, data_string):
        """Adds a data string (hash) to the tree and recalculates root."""
        leaf_hash = self._hash(data_string)
        self.leaves.append(leaf_hash)
        self.root = self._build_tree(self.leaves)
        return self.root

    def _hash(self, data):
        """SHA-256 helper"""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _build_tree(self, leaves):
        """Recursively builds the Merkle Tree root"""
        if not leaves:
            return None
        if len(leaves) == 1:
            return leaves[0]

        new_level = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i+1] if i+1 < len(leaves) else left # Duplicate last if odd
            combined = left + right
            new_level.append(self._hash(combined))
        
        return self._build_tree(new_level)

    def get_root(self):
        return self.root

class DigitalSigner:
    """
    Layer 2 Security: Digital Signatures
    Signs document data using RSA Private Key.
    """
    def __init__(self):
        # In a real app, load from secure storage (HSM/Vault). 
        # Here we generate a new keypair on init (or could load from file).
        # For persistence across restarts, we should save/load these keys.
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.public_key = self.private_key.public_key()

    def get_public_key_pem(self):
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

    def sign_data(self, data_dict):
        """Signs a dictionary of data."""
        data_str = json.dumps(data_dict, sort_keys=True)
        signature = self.private_key.sign(
            data_str.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

    def verify_signature(self, data_dict, signature_b64, public_key_pem=None):
        """Verifies a signature."""
        try:
            data_str = json.dumps(data_dict, sort_keys=True)
            signature = base64.b64decode(signature_b64)
            
            pub_key = self.public_key
            if public_key_pem:
                 pub_key = serialization.load_pem_public_key(
                    public_key_pem.encode('utf-8')
                )

            pub_key.verify(
                signature,
                data_str.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            print(f"Verification Failed: {e}")
            return False

# Global Instances (Singleton-ish for the server)
# In production, state would be in DB/Redis
merkle_tree = MerkleTree()
signer = DigitalSigner()
