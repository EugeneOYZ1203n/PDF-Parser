"""The optional deep backends need torch (the `ml` extra); without it their tests are not collected."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]
