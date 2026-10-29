Command-line interface
======================

Installing ``tensorial`` puts a ``tensorial`` command on your ``PATH``
(entry point :func:`tensorial.reaxkit.cli.main_cli`). It is a thin wrapper: it
dispatches to the Hydra-driven training and evaluation entry points, or copies
one of the bundled example configurations so you can run it straight away.

.. contents::
   :local:

.. code-block:: shell

   tensorial <command> [options]

The available commands are:

- ``train`` — fit a model from a Hydra config.
- ``predict`` — evaluate a trained checkpoint.
- ``examples`` — list or copy the bundled example configurations.

``tensorial train``
-------------------

Train a model described by a Hydra config file. ``-i`` / ``--input`` points at
the config; it defaults to ``configs/train.yaml``:

.. code-block:: shell

   tensorial train -i train.yaml
   tensorial train -i configs/my_exp.yaml

Within the config's directory, Hydra composes the file as the primary config,
so any other YAML groups next to it (``data/``, ``model/``, ``trainer/``, ...)
can be selected or overridden on the command line exactly as with a normal
Hydra app:

.. code-block:: shell

   tensorial train -i train.yaml model=mace data=qm9
   tensorial train -i train.yaml train.max_epochs=50

``-i`` is expected to be a file. If it is a directory, ``predict`` resolves the
default ``config.yaml`` inside it (see below); ``train`` does not.

``tensorial predict``
---------------------

Evaluate a trained model. It takes the same ``-i`` / ``--input`` option,
defaulting to ``configs/eval.yaml``, and forwards everything else to Hydra:

.. code-block:: shell

   tensorial predict -i eval.yaml ckpt_path=/path/to/checkpoint

Unlike ``train``, ``-i`` may also be a *directory*, in which case the file
:data:`tensorial.reaxkit.config.DEFAULT_CONFIG_FILE` (``config.yaml``) is loaded
from it:

.. code-block:: shell

   tensorial predict -i checkpoints/my_run/

An evaluation config normally needs a checkpoint path, e.g. via the
``ckpt_path`` override above.

``tensorial examples``
----------------------

The package bundles ready-to-run example configurations (currently the
``mlip`` example, a machine-learning interatomic potential). This command lists
them and copies one into your working directory so you can run it unmodified.

List the available examples:

.. code-block:: shell

   tensorial examples list
   tensorial examples            # 'list' is the default

Copy an example into the current directory (or a destination you give):

.. code-block:: shell

   tensorial examples init mlip
   tensorial examples init mlip.si path/to/dest

The ``init`` keyword is optional, so ``tensorial examples mlip.si`` is
equivalent to ``tensorial examples init mlip.si``.

Variants
--------

An example can expose **variants** through its ``data`` Hydra group: the
example differs from its siblings only in which dataset it trains on. The
``mlip`` example ships ``si``, ``qm9``, and ``mad`` variants, so it appears in
the listing as:

.. code-block:: text

   Available examples:
     - mlip.mad
     - mlip.qm9
     - mlip.si

Copying a variant (``mlip.si``) rewrites the ``data`` default in the copied
``train.yaml`` to that choice and copies only that variant's dataset files next
to it — not the sibling datasets. A bare name (``mlip``) copies the example
with its default ``data`` choice instead.

Each variant file may declare the extra files it needs with a
``# requires:`` comment; those are copied along with it:

.. code-block:: yaml

   # requires: sitraj.xyz
   _target_: tensorial.gcnn.data.ase_graph_module_random_split
   path: ${paths.data_dir}/sitraj.xyz

So ``tensorial examples init mlip.si`` produces an ``mlip-si/`` directory
containing ``train.yaml``, ``eval.yaml``, and the other groups, plus
``data/si.yaml`` and its ``data/sitraj.xyz``. From there:

.. code-block:: shell

   cd mlip-si
   tensorial train -i train.yaml

.. seealso::

   - :doc:`quickstart <quickstart>` — installing and the config-driven training
     pattern.
   - :doc:`mlip_guide <mlip_guide>` — what the bundled ``mlip`` example
     actually trains.
   - :mod:`tensorial.reaxkit` — the training, evaluation, and config API the
     command wraps.
