from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def disable_live_weather_globally():
    from app.environment import _DEFAULT_ENV_SERVICE
    orig = _DEFAULT_ENV_SERVICE.enable_live
    _DEFAULT_ENV_SERVICE.enable_live = False
    yield
    _DEFAULT_ENV_SERVICE.enable_live = orig
