import os

def is_chinet_verbose():
    """Check if verbose logging is enabled."""
    return os.getenv("chinet_VERBOSE", "0").lower() in ("1", "true", "yes")
