si: E(3)-equivariant silicon MLIP example
=========================================

An E(3)-equivariant interatomic potential for silicon, trained on the
``sitraj.xyz`` atomic trajectory (per-atom energies, forces and stresses)
with a NequIP-style backbone.

Data
----
The default configuration reads ``${paths.data_dir}/sitraj.xyz``
(``./data/sitraj.xyz`` relative to where you run), so place that file in a
``data/`` directory next to your working directory, or point at another
file on the command line::

    data.dataloader.path=/path/to/trajectory.extxyz

Benchmark datasets
------------------
The ``data/`` directory also has configurations for standard benchmarks, which
are downloaded on first use and use the published train/test splits:

- ``data=mad``: the Massive Atomic Diversity dataset (crystals, surfaces, molecules)
- ``data=rmd17``: revised MD17 small molecules; pick one with ``data.molecule=``
  and one of the five standard splits with ``data.split=``
- ``data=3bpa``: 3BPA, trained at 300 K; override ``data.test=`` to test at 600 K
  or 1200 K

Run
---
To train::

    tensorial train -i train.yaml

To evaluate a trained model on the test set, point ``predict`` at the training run's
directory, its checkpoint and its saved config::

    tensorial predict -i logs/train/<run> ckpt_path=logs/train/<run>/checkpoints/last.ckpt \
        +config_path=logs/train/<run>/config.yaml test=true

For a quick check that everything runs (one epoch on a few structures, then testing)::

    tensorial train -i train.yaml debug=smoke

You can select an alternative model or override any option on the
command line, e.g. ``tensorial train -i train.yaml model=mace`` or
``data=ase ase_data=/path/to/file.xyz``.
