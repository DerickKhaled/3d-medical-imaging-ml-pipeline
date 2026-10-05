# Notes on regulated medical software

First, to be clear: this project is not compliant with any standard, and I don't claim it
is. Compliance needs: a quality system, written processes, risk
management, clinical validation and documentation.

What code *can* do is make that work much easier. If a system was built without
traceability, adding it later is painful. So I built it in from the start.

## What the project already does

**Traceability.** Every prediction can be traced back to the scan, model, training run,
dataset and settings that produced it (`src/lineage/trace.py`). The trace re-checks the file
hashes, so it notices if anything was edited afterwards.

**Reproducibility.** Fixed seeds, deterministic PyTorch settings, and augmentation that
depends only on the seed, epoch and sample. A test trains twice and checks that the weights
are identical. On the real data, a restarted run gave exactly the same first-epoch loss.

**Dataset versioning.** The manifest lists every scan with its hash and split. The dataset
version is a hash of all of that. If one scan changes, the version changes.

**Model versioning.** When a model is registered, the checkpoint is copied and its hash is
stored. Every time it is loaded, the hash is checked. A changed file is refused.

**Controlled releases.** A model can only go from candidate to validated if its test results
pass the rules in `configs/release.yaml`. Every status change needs a written reason, and the
registry keeps who did it and when. Only one model is in production at a time.

**Testing.** 61 automated tests run on every push (GitHub Actions). They include the tricky
cases: broken scans, empty masks, geometry round trips, and tampered files.

**Research vs. production.** Training runs live in `artifacts/experiments/` and can be as messy
as needed. Only registered models that passed the release rules are used for inference, and
each inference record notes the model's status at the time.


