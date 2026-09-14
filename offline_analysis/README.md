# Offline analysis tools

This directory is intentionally outside the acquisition runtime. Nothing in
`session_recorder.py` or `run_experiment.py` imports or invokes these modules.

- `epoch_builder.py` creates legacy analysis windows when explicitly run.
- `preprocess_eeg.py` creates a separate filtered copy when explicitly run.

These tools never define whether acquisition succeeded and never overwrite
`eeg.csv`, `eeg_raw.bin`, or `events.csv`.
