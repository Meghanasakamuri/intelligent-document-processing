# backend/app package
import sys

# Safe compatibility fallback for Windows environments where Application Control (Smart App Control)
# blocks unsigned compiled C-extensions (_regex.cp313-win_amd64.pyd)
try:
    import regex  # noqa: F401
except ImportError:
    import re
    sys.modules["regex"] = re

