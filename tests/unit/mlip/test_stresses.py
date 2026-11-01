from typing import cast

import jax.numpy as jnp
import jraph

from tensorial import mlip


def test_calc_stresses_skips_graphs_without_cell():
    """Molecules have no unit cell, so there is no stress to calculate."""
    graph = jraph.GraphsTuple(
        nodes={"positions": jnp.zeros((2, 3))},
        edges={},
        receivers=jnp.array([0]),
        senders=jnp.array([1]),
        globals={},
        n_node=jnp.array([2]),
        n_edge=jnp.array([1]),
    )

    def energy_fn(g: jraph.GraphsTuple) -> jraph.GraphsTuple:
        return g._replace(globals={"predicted_energy": jnp.array([1.0])})

    out = cast(jraph.GraphsTuple, mlip.CalcStresses(energy_fn=energy_fn).apply({}, graph))
    # The energy is still evaluated, but there are no virials or stresses
    assert set(cast(dict, out.globals)) == {"predicted_energy"}
