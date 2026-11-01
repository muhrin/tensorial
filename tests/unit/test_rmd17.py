from typing import cast

import jraph
import numpy as np
import pytest
import reax

from tensorial.datasets import Rmd17, rmd17_graph_module
from tensorial.datasets.rmd17 import KCAL_PER_MOL_TO_EV, to_ase
from tensorial.gcnn.atomic import keys

N_CONFS = 20


def _sizes(datasets) -> tuple[int, int]:
    return len(datasets.get("train") or []), len(datasets.get("test") or [])


@pytest.fixture
def rmd17_dir(tmp_path):
    """A tiny, fake rMD17 ``ethanol`` dataset with the real file layout."""
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "rmd17"
    data_dir.mkdir()
    np.savez(
        data_dir / "rmd17_ethanol.npz",
        nuclear_charges=np.array([6, 6, 8, 1, 1, 1, 1, 1, 1]),
        coords=rng.uniform(-1.5, 1.5, size=(N_CONFS, 9, 3)),
        energies=np.arange(N_CONFS, dtype=float) - 97000.0,
        forces=rng.normal(size=(N_CONFS, 9, 3)),
    )
    for split in range(1, 6):
        np.savetxt(data_dir / f"index_train_{split:02d}.csv", np.arange(0, 10), fmt="%d")
        np.savetxt(data_dir / f"index_test_{split:02d}.csv", np.arange(10, 15), fmt="%d")
    return tmp_path


def test_fetch_uses_split_indices(rmd17_dir):
    datasets = Rmd17("ethanol", data_dir=rmd17_dir, download=False).fetch()
    assert set(datasets) == {"train", "test"}
    assert _sizes(datasets) == (10, 5)


def test_to_ase_converts_units(rmd17_dir):
    npz = np.load(rmd17_dir / "rmd17" / "rmd17_ethanol.npz")
    (atoms,) = to_ase(npz, np.array([12]))
    assert atoms.get_chemical_formula() == "C2H6O"
    assert not atoms.pbc.any()
    np.testing.assert_allclose(atoms.positions, npz["coords"][12])
    assert atoms.get_potential_energy() == pytest.approx(npz["energies"][12] * KCAL_PER_MOL_TO_EV)
    np.testing.assert_allclose(atoms.get_forces(), npz["forces"][12] * KCAL_PER_MOL_TO_EV)


def test_fetch_as_graphs(rmd17_dir):
    datasets = Rmd17(
        "ethanol",
        as_graphs={
            "r_max": 5.0,
            "atom_include_keys": ["numbers", "forces"],
            "global_include_keys": ["energy"],
        },
        data_dir=rmd17_dir,
        download=False,
    ).fetch()
    graph = cast(list, datasets.get("train"))[0]
    assert isinstance(graph, jraph.GraphsTuple)
    assert keys.FORCES in cast(dict, graph.nodes)
    assert keys.TOTAL_ENERGY in cast(dict, graph.globals)


def test_graph_module_holds_out_validation(rmd17_dir):
    dm = rmd17_graph_module(
        "ethanol", {"r_max": 5.0}, val_from_train=2, data_dir=rmd17_dir, batch_size=2
    )
    dm.prepare_data()
    dm.setup(reax.Engine(logger=False))
    assert _sizes({"train": dm.data_train, "test": dm.data_test}) == (8, 5)
    assert len(dm.data_val or []) == 2


def test_missing_file_without_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="download=False"):
        Rmd17("ethanol", data_dir=tmp_path, download=False).fetch()


def test_invalid_arguments():
    with pytest.raises(ValueError, match="Unknown rMD17 molecule"):
        Rmd17("water")
    with pytest.raises(ValueError, match="split must be between"):
        Rmd17("ethanol", split=6)


def test_limit(rmd17_dir):
    datasets = Rmd17("ethanol", data_dir=rmd17_dir, download=False, limit=3).fetch()
    assert _sizes(datasets) == (3, 3)


def test_downloads_missing_files(rmd17_dir, tmp_path, monkeypatch):
    """Missing files are fetched from their figshare URLs (the network call is faked here)."""
    from tensorial.datasets import rmd17

    source = rmd17_dir / "rmd17"
    urls_by_file = {}

    def fake_download(url, target):
        urls_by_file[target.name] = url
        target.write_bytes((source / target.name).read_bytes())

    monkeypatch.setattr(rmd17.urls, "_download", fake_download)
    empty_dir = tmp_path / "empty"
    datasets = Rmd17("ethanol", data_dir=empty_dir).fetch()

    assert _sizes(datasets) == (10, 5)
    assert urls_by_file == {
        "rmd17_ethanol.npz": "https://ndownloader.figshare.com/files/62265733",
        "index_train_01.csv": "https://ndownloader.figshare.com/files/62265793",
        "index_test_01.csv": "https://ndownloader.figshare.com/files/62265781",
    }
