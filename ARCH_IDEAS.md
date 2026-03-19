# Architecture Ideas

## Qt Action Router - Intermediate Layer

### Problem

Qt actions in `main.py` are connected directly to handler methods:
```python
self.actionAdd_fit.triggered.connect(self.onAddFit)  # → onAddFit() → service.fit_add()
self.actionSave_Project.triggered.connect(self.onSaveProject)
```

- 100+ Qt action connections scattered in `main.py`
- Each handler method does work AND sometimes calls action_dispatcher  
- No central place to see what Qt triggers what action
- Hard to add shortcuts, logging, or undo/redo consistently

### Solution: QtActionRouter

A thin intermediate layer that routes Qt signals → actions:

```python
class QtActionRouter:
    """Routes Qt actions to the action system."""
    
    def connect(self, qt_action, action_name: str, payload_fn=None):
        """Connect Qt action to action dispatcher."""
        def handler():
            payload = payload_fn() if payload_fn else {}
            dispatch(action_name, payload)
        
        qt_action.triggered.connect(handler)
```

### Usage in main.py:
```python
router = QtActionRouter(self)

router.connect(self.actionAdd_fit, "fit_add", lambda: {
    "dataset_indices": self.dataset_selector.selected_curve_index
})

router.connect(self.actionSave_Project, "project_save", lambda: {
    "project_name": self.current_project_name
})

router.connect(self.actionLoad_Data, "dataset_add", lambda: {})
```

### Benefits

1. **Centralized routing** - All Qt → action mappings in one place
2. **Consistent shortcuts** - Can set shortcuts in router, not each handler  
3. **Auto-logging** - Router can log all actions automatically
4. **Undo/redo** - Router can check action replayability
5. **Less code** - No need for 50+ handler methods in MainWindow
6. **Testable** - Can test routing without Qt

### Implementation

1. **Create `chisurf/gui/qt_action_router.py`** (~50 lines)

2. **Create simplified `dispatch()` function** - routes action_name → service

3. **Migrate one menu at a time**:
   - File menu (Save, Load, Project)
   - Edit menu (Undo, Redo) 
   - Fit menu
   - Data menu

4. **Remove handler methods** from MainWindow as you go

### Expected Reduction

| Metric | Before | After |
|--------|--------|-------|
| Handler methods in MainWindow | ~50 | 0 |
| Lines in MainWindow | ~5000 | ~4500 |
| ActionController | Required | Can simplify/delete |

---

## Simplify MVC/MCP System

### Current Bloat

- **ActionSpec**: 56 action registrations in `runtime/actions.py`
- **ActionController**: 497 lines with 40 handler methods (just thin wrappers)
- **Services**: 6 files with manual `record_action()` calls
- **MCP Tools**: Manual `@mcp.tool` definitions duplicating services

**Total**: ~1500+ lines across 3-4 layers

### Proposed: Use @action Decorator

```python
# BEFORE: 3 files to add one action
# runtime/actions.py: ActionSpec("model_add_component", schema={"component_name": str})
# controllers/action_controller.py: def _handle_model_add_component(...)
# controllers/services/model_service.py: def add_model_component(...): record_action(...)

# AFTER: 1 decorator does everything
@action("model_add_component", schema={"component_name": str})
def add_model_component(component_name: str, fit_index: int = 0):
    from chisurf.macros import model as model_macros
    model_macros.add_component(component_name, fit_idx=fit_index)
```

The decorator:
- Auto-registers action in catalog
- Auto-records to history
- Works for MCP without duplication

### Remove ActionController Entirely

Delete `controllers/action_controller.py` (497 lines of pure pass-through).

### Auto-Generate MCP Tools

```python
def register_mcp_tools(mcp):
    for name, fn in action_registry.actions.items():
        @mcp.tool
        def tool(**kwargs):
            return dispatch(name, kwargs)
        tool.__name__ = name.replace("_", ".")  # MCP style
```

### Expected Reduction

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| ActionController | 497 lines | 0 | 100% |
| ActionSpec boilerplate | ~200 lines | ~50 | 75% |
| Service boilerplate | ~500 lines | ~200 | 60% |
| MCP manual tools | ~300 lines | ~50 | 83% |
| Total | ~1500+ | ~300 | ~80% |
