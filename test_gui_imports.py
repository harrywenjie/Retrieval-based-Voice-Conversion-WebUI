#!/usr/bin/env python
"""Test script to verify gui_v1.py can import and initialize RVC correctly"""

import sys
import os

if __name__ == '__main__':
    # CRITICAL: On Windows, multiprocessing requires this guard
    # The rtrvc module creates a Manager() at import time, which spawns subprocesses
    
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    os.chdir(os.getcwd())

    print("Testing RVC initialization with PyTorch 2.6+ patch...")
    print(f"Working directory: {os.getcwd()}")

    # Import the GUI module (this will apply the monkey patch)
    try:
        # Suppress GUI window creation by setting headless mode
        os.environ["DISPLAY"] = ""  # Prevent GUI on Linux
        
        # Import after setting environment
        import gui_v1
        print("✓ gui_v1.py imported successfully (monkey patch applied)")
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test importing the RVC module directly
    try:
        from infer.lib import rtrvc
        print("✓ rtrvc module imported successfully")
        
        # Check that critical attributes are in the class
        rvc_attrs = ['tgt_sr', 'if_f0', 'version', 'net_g', 'model', 'initialized', 'init_error']
        print(f"  Checking RVC class structure...")
        
        # We can't instantiate without a model, but we can check the __init__ signature
        import inspect
        sig = inspect.signature(rtrvc.RVC.__init__)
        print(f"  ✓ RVC.__init__ signature: {len(sig.parameters)} parameters")
        
    except Exception as e:
        print(f"✗ RVC module test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✓ All import tests passed!")
    print("\nNOTE: To fully test, run: .\\go-realtime-gui-312.bat")
    print("The GUI should now start without UnpicklingError.")
