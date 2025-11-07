# AttributeError Fix - World Consistency Critic

## Bug Report

**Error**: 
```python
AttributeError: 'NoneType' object has no attribute 'lower'
```

**Location**: 
`world_consistency_critic.py`, line 410 in `_check_contradictions()`

**Traceback**:
```python
/kaggle/input/director-llm-critics/world_consistency_critic.py in _check_contradictions(self, claimed_updates)
    408         for obj_name, obj_state in self.tracker.objects.items():
    409             if obj_name in response_lower or obj_name in str(claimed_updates.get('objects', [])).lower():
--> 410                 current_state = obj_state.get('state', '').lower()
    411                 current_location = obj_state.get('location', '').lower()
```

## Root Cause

When `self.tracker.update_object()` is called with `state=None` or `location=None`, the dictionary stores `None` as the value:

```python
# Example:
self.tracker.objects = {
    'door': {
        'state': None,      # ← This causes the problem
        'location': None,   # ← This too
        'properties': {}
    }
}
```

When we later do:
```python
current_state = obj_state.get('state', '').lower()
```

The `.get('state', '')` returns `None` (the actual value) instead of the default `''`, because the key exists but its value is `None`. Then calling `.lower()` on `None` raises `AttributeError`.

## The Fix

**Before** (broken):
```python
current_state = obj_state.get('state', '').lower()
current_location = obj_state.get('location', '').lower()
```

**After** (fixed):
```python
current_state = (obj_state.get('state') or '').lower()
current_location = (obj_state.get('location') or '').lower()
```

### Why This Works

1. `obj_state.get('state')` returns the value (could be `None`, a string, or nothing)
2. `or ''` provides a fallback: if the left side is `None` or falsy, use `''`
3. `.lower()` is now called on a guaranteed string (either the value or `''`)

### Truth Table

| `get('state')` value | `or ''` result | `.lower()` works? |
|---------------------|----------------|-------------------|
| `None` | `''` | ✅ Yes |
| `'locked'` | `'locked'` | ✅ Yes |
| `''` | `''` | ✅ Yes |
| (key missing) | `''` | ✅ Yes |

## When This Bug Occurs

This happens when:

1. An object is mentioned in text but no explicit state is detected:
   ```python
   text = "I use the rusty key to unlock the ancient door"
   # Extracts: object='key', object='door'
   # But state detection might fail → state=None
   ```

2. The state detection logic doesn't find keywords:
   ```python
   if 'locked' in obj_context_lower:
       state = 'locked'
   # If no keywords match, state remains None
   ```

3. Objects are added without state information:
   ```python
   self.tracker.update_object('key', state=None, location=None)
   # Stores None explicitly
   ```

## Additional Defensive Fixes

We could also fix this at the source by never storing `None`:

```python
def update_object(self, obj: str, location: str = None, state: str = None, properties: Dict = None):
    """Update object state"""
    if obj not in self.objects:
        self.objects[obj] = {
            'location': location or '',  # ← Default to empty string
            'state': state or '',        # ← Default to empty string
            'properties': properties or {}
        }
```

But the current fix (using `or ''` when reading) is more defensive and handles edge cases better.

## Testing

After this fix, the following should work without errors:

```python
critic = WorldConsistencyCritic()
critic.tracker.update_object('door', state=None, location=None)
critic.update_world_state("I use the rusty key to unlock the ancient door")
score = critic.score("The key turns with a satisfying click. The ancient door swings open.")
# Should return a valid score without AttributeError
```

## Files Changed

1. **`world consistency critic/world_consistency_critic.py`**
   - Line 410: `current_state = (obj_state.get('state') or '').lower()`
   - Line 411: `current_location = (obj_state.get('location') or '').lower()`

## Status

✅ **FIXED** - The critic can now handle `None` values gracefully without raising `AttributeError`.
