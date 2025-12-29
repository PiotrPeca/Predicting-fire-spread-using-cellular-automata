import sys
from pathlib import Path


def pytest_configure() -> None:
    """Ensure `src/` is on sys.path so tests can import `fire_spread.*`.

    This repo uses the common `src/` layout but is not necessarily installed as a package
    in the active environment.
    """

    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
