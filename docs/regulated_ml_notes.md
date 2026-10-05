# Engineering for regulated medical ML: notes on this architecture

> **This project is not compliant with any standard and makes no regulatory claim.**
> This engineering architecture can support a regulated development process, but
> compliance additionally requires organizational processes, documentation,
> validation, risk management and quality-system controls.

The point of these notes is to show *which engineering decisions make a
regulated process cheaper and safer later*. Retrofitting traceability into an
ML system that was built without it is usually more expensive than building it
in from the start.

## Principles and where they live in the code

| Principle | What it means in practice | Where in this repo |
|---|---|---|
| **Traceability** | Every output can be followed back to its inputs, code and decisions | `src/lineage/trace.py` walks prediction → model → evaluation → experiment → dataset → preprocessing → source scan, and **re-hashes** every link |
| **Reproducibility** | Same inputs + same code + same config → same result | fixed seeds, deterministic kernels, seed-derived augmentation (`src/training/dataset.py`), content-hashed preprocessing; `tests/test_model_training.py::test_training_is_reproducible` asserts identical weights across two runs |
| **Dataset lineage** | Know exactly which scans trained/validated/tested a model | `src/data/manifest.py`: `dataset_version` = hash of scans, labels and split; hash-based split never moves a test scan into training when data is added |
| **Model lineage** | Know exactly which checkpoint is deployed and where it came from | registry copies the checkpoint, stores its SHA-256, and verifies it on every load (`src/models/checkpoint.py`) |
| **Controlled releases** | A model reaches production only through an explicit, evidenced decision | `src/registry/model_registry.py`: candidate → validated (release gate from `configs/release.yaml`) → production; written reason required; history of who/when/why |
| **Change control** | Any change produces a new, distinguishable version | versions are immutable; preprocessing version is derived from config content; a changed scan changes the dataset version |
| **Software verification** | Each unit does what it specifies | `tests/`: 60+ tests incl. metric edge cases, geometry round-trips, malformed input, tamper detection |
| **Validation evidence** | Performance claims are measured on held-out data with the shipped code path | `src/evaluation/evaluate.py` runs the *production* `segment()` function on the test split, in original scan geometry; the evaluation file hash is pinned in the registry |
| **Risk management** | Known failure modes have controls | input validation and limits shipped with the model; inference refuses tampered checkpoints; trace warns when a "test" scan was actually in training data |
| **Separation of research and production** | Experiments can be messy; releases cannot | experiments live in `artifacts/experiments/`; only registered, gated, hash-pinned models are served; inference records the model's status at inference time |
| **Auditability** | A reviewer can answer questions from records, not memory | plain JSON records, atomic writes, no hidden state; `git_commit` and dirty-tree flag in every experiment |

## How the standards relate (high level)

**IEC 62304 (medical device software life cycle).** Requires a defined
development process, software requirements, architecture, unit/integration
verification, configuration management and problem resolution, scaled by
software safety class. This repo shows the *technical* side of configuration
management (versioned data, configs, models, code commit per artifact) and
verification (automated tests in CI). It does not include requirements
specifications, a documented software development plan, SOUP management or
formal problem resolution, which a real product needs.

**ISO 14971 (risk management for medical devices).** Requires identifying
hazards, estimating and controlling risks, and verifying the controls. In ML,
typical hazards are distribution shift, wrong input modality/geometry, silent
model substitution and data leakage. This repo implements some *controls*
(input limits, checksum-verified models, leakage warning, release gates) but
not the risk analysis itself (hazard identification, severity/probability,
residual risk acceptance).

**ISO 13485 (quality management systems).** The organizational framework:
document control, design controls, supplier management, CAPA, training
records. Code cannot provide this; it can make design-control evidence (design
inputs → outputs → verification → validation) easy to produce and audit.

**EU AI Act.** Most AI medical device software that requires a notified body
under the MDR is classified as *high-risk* AI. Obligations include data
governance (relevance, representativeness, error handling), technical
documentation, record-keeping/logging, transparency, human oversight, accuracy
and robustness. The dataset manifest, evaluation records, inference records and
lineage trace are the kind of technical artefacts these obligations draw on.
Note: application dates for high-risk obligations have been subject to change;
check the current legal text before relying on any date.

## What a real product would add

- Requirements and a traceability matrix from requirement → design → test
- Formal risk analysis (ISO 14971) linking hazards to the controls above
- Subgroup / stratified performance analysis (scanner vendor, field strength,
  age, pathology), failure-case review, and clinically meaningful acceptance criteria
- Post-market monitoring: drift detection on input statistics and outputs
- Access control and electronic signatures on promotions (21 CFR Part 11 style)
- An append-only, tamper-evident store for records (instead of local JSON files)
- Cybersecurity (IEC 81001-5-1) and SOUP/dependency management with pinned versions
- Clinical evaluation and usability engineering (IEC 62366) for the viewer
