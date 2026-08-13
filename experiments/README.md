Experiment recording guide

How to record experiments for reproducibility and portfolio tracking

- Keep large artifacts (checkpoints, model binaries, videos) in `outputs/` (ignored).
- Commit only small metadata files in `experiments/` that reference the commit, config, and artifact location.
- Use the provided `template.yaml` to record each run. Name files like `2026-08-13_act-v1.yaml`.
- Example workflow:
  1. Create a branch: `git checkout -b exp/act-seed42`
  2. Edit your config file under `configs/` or `Ubtech_sim/config/`.
  3. Run experiment; store large outputs in `outputs/exp-name/`.
  4. Create an experiment metadata file in `experiments/` using `template.yaml`.
  5. Commit code, configs, and the metadata file; push branch and open a PR for merging.

Keep metadata minimal (YAML) and include the `commit` hash so results are reproducible.

Recommended tools:
- Use DVC or Git LFS for versioning large model files and dataset pointers.
- Use a remote object store (S3, GCS) or DVC remote to keep heavy artifacts off Git.

Contact: keep experiment notes concise and reference external storage locations rather than including large files in the repo.
