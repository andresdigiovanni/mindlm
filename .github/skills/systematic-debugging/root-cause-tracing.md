# Root Cause Tracing

## Overview

Bugs often manifest deep in the call stack. Your instinct is to fix where the error appears, but that's treating a symptom.

**Core principle:** Trace backward through the call chain until you find the original trigger, then fix at the source.

## When to Use

**Use when:**
- Error happens deep in execution (not at entry point)
- Stack trace shows long call chain
- Unclear where invalid data originated
- Need to find which test/code triggers the problem

## The Tracing Process

### 1. Observe the Symptom
```
Error: unexpected value at src/module/processor.py line 42
```

### 2. Find Immediate Cause
**What code directly causes this?**
```python
result = process_data(value)  # value is empty string here
```

### 3. Ask: What Called This?
```python
Processor.handle_request(value)
  → called by RequestHandler.dispatch()
  → called by API.post()
  → called by test at TestAPI.test_create()
```

### 4. Keep Tracing Up
**What value was passed?**
- `value = ''` (empty string!)
- Empty string passed from request body without validation
- That's the source!

### 5. Find Original Trigger
**Where did empty string come from?**
- Input validation missing at API boundary
- Request body not validated before processing

### 6. Fix at Source
**Fix where empty string entered, NOT where it crashed:**
```python
# Fix at API boundary (source)
if not value:
    raise ValueError("value cannot be empty")

# NOT at processor (symptom)
if not value:
    return default  # masks the real bug
```

## Pattern: Empty/None Values

```
Symptom: AttributeError: 'NoneType' has no attribute 'X'
Trace:   None came from → function returned None → input was None → not validated at entry
Fix at:  Entry point validation, not attribute access
```

## Pattern: Wrong Directory/Path

```
Symptom: FileNotFoundError in deeply nested function
Trace:   path = '' → came from config['path'] → config not loaded → file missing
Fix at:  Config loading, not file access
```

## Quick Checklist

1. Where does the error occur? (symptom)
2. What is the bad value at that point?
3. What function passed that bad value?
4. What passed it to that function?
5. Continue until you reach the boundary (input/config/external call)
6. Fix at the boundary
