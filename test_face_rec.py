import face_recognition
import numpy as np

print("Testing Face Recognition...")

try:
    # synthetic image: 100x100 RGB, all black
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    print(f"Created Synthetic Image: Shape={img.shape}, Dtype={img.dtype}")
    
    # Try encodings (should return empty list, but NOT crash)
    print("Running face_encodings...")
    encodings = face_recognition.face_encodings(img)
    print(f"Success! Encodings (should be empty): {encodings}")

except Exception as e:
    print(f"CRITICAL FAILURE: {e}")
