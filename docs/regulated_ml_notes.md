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

## The standards, in short

**IEC 62304** is about the software development process for medical device software:
requirements, architecture, testing, configuration management and bug handling. This project
covers some of the technical side (versioning and automated tests). It has no written
requirements, development plan or formal bug process.

**ISO 14971** is risk management: find what could go wrong, judge how bad it is, and put
controls in place. This project has some controls (input checks, verified model files, the
data leakage warning, release rules), but no real risk analysis behind them.

**ISO 13485** is the quality management system for the whole company: document control,
design reviews, suppliers, corrective actions, training. That is organisation, not code. Good
code just makes the evidence easier to produce.

**EU AI Act.** Most AI software in medical devices counts as high-risk AI. That brings rules on
data quality, technical documentation, logging, transparency, human oversight, accuracy and
robustness. Records like the manifest, evaluation results, inference logs and trace are the
kind of material those rules ask for. The dates when these rules apply have changed before, so
check the current text.

## What a real product would still need

- written requirements, and a link from each requirement to its tests
- a real risk analysis connected to the controls above
- evaluation per patient group, scanner and disease, and a review of failure cases with clinicians
- monitoring after release, to notice when incoming scans start to look different
- user accounts and signatures on model promotions
- tamper-proof storage for the records instead of local JSON files
- cybersecurity and dependency management
- clinical evaluation and usability testing for the viewer
