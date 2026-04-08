from config import Config

try:
    import weave
except Exception:
    weave = None


_WEAVE_INITIALIZED = False


def init_weave(project_name: str = "") -> bool:
    global _WEAVE_INITIALIZED

    if _WEAVE_INITIALIZED or not Config.WEAVE_ENABLED:
        return _WEAVE_INITIALIZED

    if weave is None:
        print("[Weave] weave import failed, tracing disabled.")
        return False

    target_project = project_name or Config.WEAVE_PROJECT
    try:
        weave.init(target_project)
        _WEAVE_INITIALIZED = True
    except Exception as exc:
        print(f"[Weave] init failed, tracing disabled: {exc}")
        _WEAVE_INITIALIZED = False
    return _WEAVE_INITIALIZED


def weave_op(fn):
    if weave is None:
        return fn
    try:
        return weave.op(fn)
    except Exception:
        return fn
