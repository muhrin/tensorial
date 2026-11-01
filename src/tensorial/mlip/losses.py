from .. import gcnn
from . import keys
from .keys import predicted

__all__ = ("mlip_loss",)


def mlip_loss(
    energy: bool | None = None,
    forces: bool | None = None,
    energy_weight: float = 1.0,
    force_weight: float = 20.0,
    *,
    stress: bool | None = None,
    stress_weight: float = 1.0,
    energy_per_atom: bool = True,
    loss_fn: str = "squared_error",
) -> gcnn.WeightedLoss:
    """Build the standard energy/forces/stress loss for an interatomic potential.

    Each term can be required (``True``), left out (``False``) or optional (``None``, the
    default).  An optional term is used only when the data has values for it, so the same loss
    works for, e.g., datasets of molecules (no stresses) and of crystals.  Structures whose
    stress was filled in with a placeholder (in a dataset that mixes the two) are masked out.

    Args:
        energy: whether to include the total energy term
        forces: whether to include the forces term
        energy_weight: weight of the energy term
        force_weight: weight of the forces term
        stress: whether to include the stress term
        stress_weight: weight of the stress term
        energy_per_atom: compare energies per atom rather than in total, so structures of
            different sizes contribute equally
        loss_fn: the name of the ``optax.losses`` function to apply to each term

    Returns:
        the weighted sum of the included terms
    """
    terms = [
        (
            energy,
            energy_weight,
            f"globals.{keys.TOTAL_ENERGY}",
            "n_node" if energy_per_atom else None,
        ),
        (forces, force_weight, f"nodes.{keys.FORCES}", None),
        (stress, stress_weight, f"globals.{keys.STRESS}", None),
    ]

    weights: list[float] = []
    loss_terms: list[gcnn.GraphLoss] = []
    for include, weight, field, normalise_by in terms:
        if include is False:
            continue
        root, key = field.split(".")
        weights.append(weight)
        loss_terms.append(
            gcnn.Loss(
                loss_fn,
                field,
                f"{root}.{predicted(key)}",
                optional=include is None,
                normalise_by=normalise_by,
            )
        )

    if not loss_terms:
        raise ValueError(
            "Could not create loss function because all terms (energy, forces, stress) are set to "
            "`False`"
        )

    return gcnn.WeightedLoss(loss_terms, weights)
