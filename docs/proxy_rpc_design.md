# Proxy/RPC Design Targets

## Core Principle
Zero magic. Every proxy object is an explicit, typed wrapper with well-defined
properties and methods. No `__getattr__` interception, no lazy-fetch, no dynamic
RPC method dispatch, no key aliasing.

---

## 1. Server Response Consistency

### 1.1 Single canonical key per field
No duplicate keys. Every field appears exactly once in every response.

| Field | Canonical key | Eliminated duplicate |
|---|---|---|
| Unique identifier | `uid` | `unique_identifier` ✗ |
| Fit UID in param lists | `fit_uid` | — (already single) |
| Dataset UID in fit info | `dataset_uid` | — (already single) |

### 1.2 Self-contained list responses
`list_fits()` and `list_datasets()` return **complete** data. No lazy-fetch needed.

```
fit: {
  index, uid, name, type,
  chi2, chi2r, dataset_name, dataset_uid,
  data:         { name, uid, type, filename },
  model:        { name, n_points, n_free, chi2r },
  parameters:   [ { name, value, fixed, bounds_on, linked_to, fit_uid, error_estimate } ],
}
```

```
dataset: {
  index, uid, name, type, filename, length,
}
```

Heavy payloads (curve data arrays) remain behind explicit endpoints
(`dataset__curve_data`, `fit__curve_data`).

### 1.3 Parameter representation
Every parameter dict contains exactly these fields:
```
{ name, value, fixed, bounds, bounds_on, is_linked, linked_to, error_estimate, fit_uid }
```

---

## 2. Proxy Architecture

### 2.1 Typed proxy classes (no `_ItemProxy`)

Replace the generic `_ItemProxy` with typed classes:

| Class | Wraps |
|---|---|
| `FitProxy` | A fit response dict |
| `DatasetProxy` | A dataset response dict |
| `ParameterProxy` | A parameter response dict from `parameters` list |
| `ModelProxy` | The `model` sub-dict of a fit |
| `DataProxy` | The `data` sub-dict of a fit |

### 2.2 No `__getattr__` magic

Every readable attribute is either:
- A direct key lookup on the underlying dict (for data fields), OR
- An explicit `@property` (for computed fields)

No `__getattr__` interception. If an attribute is not on the dict, it is
not available — no lazy-fetch fallback, no dynamic resolution.

### 2.3 Explicit mutation methods

Every mutation is an explicit method call. No `__setattr__` interception.

```python
# Instead of: p.value = 3.5      (magic setattr → RPC)
parameter.set_value(3.5)

# Instead of: p.fixed = True      (magic setattr → RPC)
parameter.set_fixed(True)

# Instead of: p.link = other      (magic setattr → RPC)
parameter.link_to(other)

# Instead of: fit.fit_range = (0, 100)   (magic setattr → RPC)
fit.set_fit_range(0, 100)
```

### 2.4 Explicit RPC action methods

Every server-callable action is an explicit method. No `_RPC_METHODS` dispatch.

```python
class FitProxy:
    def run(self) -> dict: ...
    def save(self, filename: str, file_type: str = "csv", **kw) -> dict: ...
    def update(self) -> dict: ...
    def set_result_idx(self, idx: int) -> dict: ...
    def set_dataset(self, dataset_index: int = None, dataset_uid: str = None) -> dict: ...
    def model_finalize(self) -> dict: ...
    def model_set_parse_function(self, function_name: str) -> dict: ...
```

```python
class ParameterProxy:
    def set_value(self, value: float) -> dict: ...
    def set_fixed(self, fixed: bool) -> dict: ...
    def set_bounds(self, bounds: tuple) -> dict: ...
    def set_bounds_on(self, bounds_on: bool) -> dict: ...
    def link_to(self, target_parameter_name: str, **kw) -> dict: ...
    def unlink(self) -> dict: ...
```

### 2.5 Explicit curve data fetch

```python
class DatasetProxy:
    def curve_data(self) -> dict:
        """Return {x, y, ex, ey} arrays from server."""
        ...
```

```python
class FitProxy:
    @property
    def curve_data(self) -> dict:
        """Return {x, y, fx, fy, residuals, fit_x, fit_y} arrays from server."""
        ...
```

---

## 3. List Proxies

### 3.1 Common interface

All list proxies implement:
```
__getitem__, __iter__, __len__, __contains__, index, pop, clear
```

`__getitem__` returns the correct typed proxy (FitProxy, DatasetProxy, etc.).

```python
class ProxyFitList:
    def __getitem__(self, idx) -> FitProxy: ...
    def __iter__(self) -> Iterator[FitProxy]: ...

class ProxyDatasetList:
    def __getitem__(self, idx) -> DatasetProxy: ...
    def __iter__(self) -> Iterator[DatasetProxy]: ...
```

### 3.2 No `append()` on proxies

`append()` is a no-op (datasets/fits must be created server-side via RPC).

---

## 4. Client API

### 4.1 Method naming

All client methods follow `{noun}__{verb}` snake_case:
```
fit__create, fit__run, fit__list, fit__remove, fit__clear
dataset__add, dataset__list, dataset__remove, dataset__clear
parameter__set_value, parameter__set_fixed, parameter__link
session__describe, session__clear, session__restore
meta__ping, meta__methods
```

### 4.2 Every method has typed parameters

```python
def fit__run(self, fit_index: int = None, fit_uid: str = None) -> dict: ...
def parameter__set_value(self, parameter_name: str, value: float,
                         fit_index: int = None, fit_uid: str = None) -> dict: ...
```

---

## 5. Implementation Plan

### Phase A: Clean server responses
1. Remove `unique_identifier` from all fit/dataset responses
2. Add `parameters` list to `list_fits()` response (inline parameter data)
3. Add `model` and `data` sub-dicts to `list_fits()` response
4. Ensure self-consistency: every key is present where needed

### Phase B: Rewrite proxy classes
1. Remove `_ItemProxy` entirely
2. Create `FitProxy`, `DatasetProxy`, `ParameterProxy`, `ModelProxy`, `DataProxy`
3. Each has explicit properties and methods (no `__getattr__`)
4. `ProxyFitList.__getitem__` returns `FitProxy`, etc.
5. Remove `_RPC_METHODS`, `_PARAM_SETTER_NAMES`, `_FIT_SETTER_NAMES`,
   `_POS_ARGS`, `_KEY_ALIASES`, `_CURVE_DATA_ATTRS`, `_NOOP_METHODS`
6. Remove `__setattr__` magic
7. Remove lazy-fetch logic
8. Keep list operations (`pop`, `clear`, `index`, `__contains__`) unchanged

### Phase C: Update tests
1. Replace all `_ItemProxy` references with typed proxies
2. Update assertions for changed server response keys (`uid` not `unique_identifier`)
3. Update setattr-based mutation tests to use explicit methods
4. Update RPC-dispatch tests to use explicit methods
5. Update lazy-fetch tests (no longer needed)

### Phase D: Documentation
1. This document defines the target
2. API reference for each proxy class and client method
