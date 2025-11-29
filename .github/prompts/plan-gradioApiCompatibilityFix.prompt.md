# Plan: Fix Gradio API Compatibility Error

## Problem

The error occurs because you're using a newer version of Gradio (likely 4.x or 5.x) which removed the `concurrency_count` parameter from the `queue()` method. The project was written for Gradio 3.34.0 but `requirements-312.txt` doesn't pin the Gradio version, causing the latest version to be installed.

Error:
```
TypeError: Blocks.queue() got an unexpected keyword argument 'concurrency_count'
```

Location: `infer-web.py`, line 1614

## Root Cause

- **Gradio 3.x (3.34.0)**: `queue(concurrency_count=511, max_size=1022)` was valid
- **Gradio 4.x+**: The `concurrency_count` parameter was removed/renamed
- **Current issue**: `requirements-312.txt` doesn't pin Gradio version, so latest version is installed

## Solution Steps

### Step 1: Update requirements-312.txt
Add `gradio==3.34.0` to pin the Gradio version for Python 3.12 compatibility

### Step 2: Update infer-web.py
Remove the deprecated `concurrency_count` parameter from both `.queue()` calls (lines 1612 and 1614), keeping only `max_size=1022`

**Line 1612** (Colab environment):
```python
# Before:
app.queue(concurrency_count=511, max_size=1022).launch(share=True)

# After:
app.queue(max_size=1022).launch(share=True)
```

**Line 1614** (Local environment):
```python
# Before:
app.queue(concurrency_count=511, max_size=1022).launch(
    server_name="0.0.0.0",
    inbrowser=not config.noautoopen,
    server_port=config.listen_port,
    quiet=True,
)

# After:
app.queue(max_size=1022).launch(
    server_name="0.0.0.0",
    inbrowser=not config.noautoopen,
    server_port=config.listen_port,
    quiet=True,
)
```

### Step 3: Verify
Confirm these are the only two `.queue()` calls in the codebase (they are)

## Further Considerations

### Gradio Version Strategy
Would you prefer to:
- **(A)** Pin to Gradio 3.34.0 in requirements-312.txt (preserves backward compatibility)
- **(B)** Update code to support Gradio 4.x+ API (modernize codebase)
- **(C)** Both - make code compatible with newer Gradio AND add version pin (most flexible)

### Testing
After the fix, test that:
1. Web UI launches successfully with Python 3.12.9
2. All functionality works as expected
3. No other Gradio-related errors occur

## Files to Modify

1. `requirements-312.txt` - Add Gradio version pin
2. `infer-web.py` - Update lines 1612 and 1614 to remove `concurrency_count` parameter
