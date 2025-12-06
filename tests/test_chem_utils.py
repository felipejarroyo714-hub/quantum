import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from chem_utils import parse_molecule_source


def test_parse_molecule_source_smiles_lenient():
    smiles, coords = parse_molecule_source("CC", strict=False)
    assert smiles is None or isinstance(smiles, str)
    if coords is not None:
        assert isinstance(coords, np.ndarray)
        assert coords.shape[1] == 3

