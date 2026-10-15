from pathlib import Path

import pytest

from hdeeg_snpm.clusters import ChannelGraph
from hdeeg_snpm.montage import get_montage

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def egi_graph():
    return ChannelGraph(get_montage("egi").neighbors)
