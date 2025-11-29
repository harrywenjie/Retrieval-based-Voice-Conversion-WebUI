#!/usr/bin/env python
"""Test script to verify PyTorch 2.6+ compatibility patch works"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Apply the monkey patch
import torch
_original_torch_load = torch.load

def _torch_load_with_weights_only_false(*args, **kwargs):
    """Wrapper for torch.load that sets weights_only=False by default"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _torch_load_with_weights_only_false

print("✓ PyTorch monkey patch applied")
print(f"  PyTorch version: {torch.__version__}")

# Try loading the HuBERT model with fairseq
try:
    import fairseq
    print("✓ Fairseq imported")
    
    hubert_path = "assets/hubert/hubert_base.pt"
    if not os.path.exists(hubert_path):
        print(f"✗ HuBERT model not found at {hubert_path}")
        sys.exit(1)
    
    print(f"✓ HuBERT model file exists: {hubert_path}")
    print("  Loading model with fairseq...")
    
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [hubert_path],
        suffix="",
    )
    
    print(f"✓ SUCCESS! Model loaded: {type(models[0])}")
    print(f"  Model has {sum(p.numel() for p in models[0].parameters())} parameters")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed! The patch is working correctly.")
