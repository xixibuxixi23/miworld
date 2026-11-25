# Training recipes

Only recipes for the currently selected lineages are kept on the main branch.
They use environment variables for paths and distributed settings so the same
commit can run on either RCZ or B200. Historical cluster launchers remain with
their immutable run artifacts and are not copied into this repository.

Before launching, record the checkpoint hash, code commit, data manifest hash,
GPU count, per-GPU batch, gradient accumulation, and output directory.

