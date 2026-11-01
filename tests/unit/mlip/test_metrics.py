import pytest
import reax


def test_metrics_plugin():
    """Check that our plugins get automatically picked up by REAX"""
    registry = reax.metrics.get_registry()

    expected = [
        "atomic/energy_per_atom_rmse",
        "atomic/energy_per_atom_mae",
        "atomic/force_rmse",
        "atomic/stress_rmse",
    ]

    for metric_name in expected:
        assert metric_name in registry


def _stress_graph(with_stress: bool, stress_mask=None):
    import jax.numpy as jnp
    import jraph

    graph_globals = {"mask": jnp.array([True, True])}
    if with_stress:
        graph_globals["stress"] = jnp.zeros((2, 3, 3))
        graph_globals["predicted_stress"] = jnp.stack(
            [jnp.full((3, 3), 2.0), jnp.full((3, 3), 10.0)]
        )
    if stress_mask is not None:
        graph_globals["stress_mask"] = jnp.array(stress_mask)
    return jraph.GraphsTuple(
        nodes={},
        edges={},
        receivers=jnp.array([], dtype=int),
        senders=jnp.array([], dtype=int),
        globals=graph_globals,
        n_node=jnp.array([0, 0]),
        n_edge=jnp.array([0, 0]),
    )


def test_stress_rmse_applies_field_mask():
    from tensorial.mlip.metrics import StressRmse

    assert float(StressRmse.create(_stress_graph(True)).compute()) == pytest.approx(
        ((4.0 + 100.0) / 2) ** 0.5
    )
    masked = StressRmse.create(_stress_graph(True, stress_mask=[True, False]))
    assert float(masked.compute()) == pytest.approx(2.0)


def test_stress_rmse_is_skipped_without_stresses():
    collection = reax.metrics.build_collection(
        {"srmse": "atomic/stress_rmse", "ermse": "atomic/energy_per_atom_rmse"}
    )
    # Stresses are optional, but energies are not
    with pytest.raises(KeyError):
        collection.create(_stress_graph(False), ignore_missing=True)

    collection = reax.metrics.build_collection({"srmse": "atomic/stress_rmse"})
    assert "srmse" not in dict(collection.create(_stress_graph(False), ignore_missing=True).items())
