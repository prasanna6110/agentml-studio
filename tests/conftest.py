import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from generate_datasets import customer_churn, house_prices  # noqa: E402

from agentml.config import Settings  # noqa: E402
from agentml.llm import MockLLM  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(workspace=tmp_path / "ws")
    s.ensure_workspace()
    return s


@pytest.fixture
def llm() -> MockLLM:
    return MockLLM()


@pytest.fixture(scope="session")
def churn_df() -> pd.DataFrame:
    return customer_churn(n=600, seed=1)


@pytest.fixture(scope="session")
def house_df() -> pd.DataFrame:
    return house_prices(n=500, seed=2)


@pytest.fixture
def churn_csv(tmp_path, churn_df) -> Path:
    p = tmp_path / "churn.csv"
    churn_df.to_csv(p, index=False)
    return p
