import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="session")
def fixture_raw(tmp_path_factory):
    from stewbench.data.fixture import write_fixture
    return write_fixture(tmp_path_factory.mktemp("raw"), n_subjects=8, seconds=24, seed=1)


@pytest.fixture(scope="session")
def windows(fixture_raw):
    from stewbench.data.preprocess import preprocess_recordings
    from stewbench.data.raw import load_stew
    from stewbench.settings import PreprocessConfig
    return preprocess_recordings(load_stew(fixture_raw, strict=False), PreprocessConfig())
