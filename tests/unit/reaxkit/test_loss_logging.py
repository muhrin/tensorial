"""The loss is logged as a raw value, so `reax` has to be told how it was reduced over the batch,
otherwise a mean gets divided by the batch size a second time when the epoch value is computed.
"""

from flax import linen
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import reax

from tensorial import reaxkit

# Ragged on purpose: 10 samples in batches of 4 gives batches of 4, 4 and 2, so weighting by batch
# size and weighting by batch are not the same thing
NUM_SAMPLES = 10
BATCH_SIZE = 4


class Squared(linen.Module):
    """A model with no parameters, so the loss is the same every epoch"""

    @linen.compact
    def __call__(self, x):
        return x**2


def mean_loss(predictions, targets):
    return optax.l2_loss(predictions, targets).mean()


def sum_loss(predictions, targets):
    return optax.l2_loss(predictions, targets).sum()


@pytest.fixture(name="dataset")
def dataset_fixture() -> np.ndarray:
    return np.random.default_rng(0).random((NUM_SAMPLES, 3))


def batches(dataset: np.ndarray) -> list[np.ndarray]:
    return [dataset[i : i + BATCH_SIZE] for i in range(0, len(dataset), BATCH_SIZE)]


def fit(dataset: np.ndarray, loss_fn, **kwargs) -> dict:
    module = reaxkit.ReaxModule(
        Squared(), loss_fn=loss_fn, optimizer=optax.adamw(learning_rate=0.01), output=None, **kwargs
    )
    module.debug = False  # `Squared` has no parameters, so there are no gradient norms to log
    trainer = reax.Trainer()
    # Inputs and targets are the same array: the loss compares the model output with its input
    loader = reax.data.ArrayLoader((dataset, dataset), batch_size=BATCH_SIZE)
    trainer.fit(module, train_dataloaders=loader, max_epochs=1)

    return trainer.logged_metrics


def test_mean_loss_is_not_scaled_by_the_batch_size(dataset):
    """The epoch loss should be the loss of the data, not the loss divided by the batch size"""
    per_batch = [float(mean_loss(batch**2, batch)) for batch in batches(dataset)]
    expected = np.average(per_batch, weights=[len(batch) for batch in batches(dataset)])

    logged = fit(dataset, mean_loss, loss_reduction="mean")["train/loss"]

    assert jnp.isclose(logged, expected, rtol=1e-5)
    # The old behaviour, for the record: roughly `batch size` times smaller
    assert not jnp.isclose(logged, sum(per_batch) / NUM_SAMPLES)


def test_sum_loss_is_unchanged(dataset):
    """A loss returning a total per batch aggregates the way it always has"""
    per_batch = [float(sum_loss(batch**2, batch)) for batch in batches(dataset)]
    expected = sum(per_batch) / NUM_SAMPLES

    logged = fit(dataset, sum_loss, loss_reduction="sum")["train/loss"]

    assert jnp.isclose(logged, expected, rtol=1e-5)


def test_plain_callable_defaults_to_sum(dataset):
    """A loss function that can't be asked how it reduces keeps `reax`'s existing convention"""
    assert fit(dataset, sum_loss)["train/loss"] == fit(dataset, sum_loss, loss_reduction="sum")[
        "train/loss"
    ]


def test_reduction_is_taken_from_the_loss_function(dataset):
    """A loss that says how it reduces doesn't need to be told twice"""

    class MeanLoss:
        reduction = "mean"

        def __call__(self, predictions, targets):
            return mean_loss(predictions, targets)

    assert (
        fit(dataset, MeanLoss())["train/loss"]
        == fit(dataset, mean_loss, loss_reduction="mean")["train/loss"]
    )
