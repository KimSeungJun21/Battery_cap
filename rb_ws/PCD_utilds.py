"""Legacy typo-compatible alias for PCD_utils.py."""

try:
    from rb_ws.PCD_utils import *  # noqa: F401,F403
    from rb_ws.PCD_utils import __all__  # noqa: F401
except ModuleNotFoundError:
    from PCD_utils import *  # noqa: F401,F403
    from PCD_utils import __all__  # noqa: F401
