from unittest.mock import patch

import jraph
import pytest

# Do not import ase here to avoid circular imports during collection
from tensorial.gcnn.data._ase import AseDataLoader


@pytest.fixture
def temp_xyz_file(tmp_path):
    import ase.build
    import ase.io

    # Create structures
    atoms = [ase.build.molecule("H2O"), ase.build.molecule("H2O")]

    # Save to temp file
    file_path = tmp_path / "test.xyz"
    ase.io.write(str(file_path), atoms)
    return str(file_path)


@pytest.fixture
def mock_atomic_conversion():
    with patch("tensorial.gcnn.data._ase.atomic.graph_from_ase") as mock_conv:
        # Return a dummy GraphsTuple
        mock_conv.return_value = jraph.GraphsTuple(
            n_node=None,
            n_edge=None,
            nodes=None,
            edges=None,
            globals=None,
            senders=None,
            receivers=None,
        )
        yield mock_conv


def test_ase_data_loader_loading(temp_xyz_file):
    # Verify it loads the correct number of items
    loader = AseDataLoader(path=temp_xyz_file)
    assert len(loader) == 2
    assert loader[0].get_chemical_formula() == "H2O"


def test_ase_data_loader_lazy_conversion(temp_xyz_file, mock_atomic_conversion):
    # Pass as_graphs to trigger lazy conversion
    # Note: AseDataLoader(..., as_graphs=...) triggers __getitem__(0) in __init__
    # to validate the arguments.

    loader = AseDataLoader(path=temp_xyz_file, as_graphs={"r_max": 3.0})

    # Due to the validation in __init__, conversion of [0] happens during init
    assert mock_atomic_conversion.call_count == 1

    # Access item 1 - should trigger conversion
    item = loader[1]

    assert isinstance(item, jraph.GraphsTuple)
    assert mock_atomic_conversion.call_count == 2
