
print("Testing imports...")
try:
    import webrtcvad
    print("SUCCESS: webrtcvad imported")
except ImportError:
    # Try importing manually if necessary, or just rely on resemblyzer
    try:
        import webrtcvad_wheels as webrtcvad
        print("SUCCESS: webrtcvad_wheels imported (as fallback logic if needed, but resemblyzer uses webrtcvad)")
        # In our case, webrtcvad-wheels installs a package named 'webrtcvad' usually, 
        # OR it installs 'webrtcvad' module. available as 'import webrtcvad'.
        # Let's check what webrtcvad points to.
    except ImportError as e:
        print(f"FAILURE: webrtcvad not found: {e}")

try:
    from resemblyzer import VoiceEncoder
    print("SUCCESS: resemblyzer imported")
    # encoder = VoiceEncoder() # This might verify model download too, but might be slow.
    # print("SUCCESS: VoiceEncoder initialized")
except ImportError as e:
    print(f"FAILURE: resemblyzer import failed: {e}")
except Exception as e:
    print(f"FAILURE: resemblyzer initialization failed: {e}")
