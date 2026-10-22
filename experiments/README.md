# Historical experiments

This folder contains source code, settings, and results from the earlier
experiments.

- `results/` contains selected historical result snapshots, including failed
  acceptance gates.
- `final_training_configuration.json` records the final pilot's training phase
  settings, source lineage, and common evaluation configuration.
- `requirements-training-v2.txt` records pinned SDK and cookbook dependencies.
- `archive/` contains historical project implementation modules outside the
  installed public package.

Account namespaces and local user paths have been normalized. Historical
programs refer to frozen research manifests, data files, checkpoints, and
temporary SDK environments from the original runs. They are source material
for inspection, not supported entry points for a new account. Some archived
modules import other historical package paths. They should be read together
with the configuration and history, rather than copied into a live workflow.

The actively supported public commands are the offline simulator demo, recorded
run replay, and release verification described in the root README. They neither
create deployments nor train models.

Model adapters and the full historical training datasets are not included.
The snapshots document the earlier runs but are not sufficient to retrain every
repair stage from a fresh checkout.
