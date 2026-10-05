# 3D Medical Imaging ML Pipeline

A small, end-to-end, **reproducible and traceable** pipeline that turns an MRI/CT
volume into a 3D segmentation and interactive surface model, and can prove where
every output came from.

> **I do not only train models. I build reproducible, traceable and maintainable ML
> systems that can move toward production.**

```
CT / MRI ─▶ validate ─▶ manifest ─▶ preprocess ─▶ PyTorch 3D U-Net ─▶ evaluate ─▶ registry
                                                                                  │ (release gate)
 new scan ─▶ inference (registered preprocessing, hash-verified model) ◀─────────┘
                │
                ├─▶ segmentation (NIfTI, in the scan's geometry) + inference record
                ├─▶ 3D meshes per structure (STL / PLY / OBJ, scanner mm coordinates)
                ├─▶ interactive viewer (3D + slice overlay)
                └─▶ lineage trace: prediction → model → evaluation → experiment → dataset → scan
```

> **Medical disclaimer.** This project is a technical demonstration only and is not
> intended for diagnosis, treatment, clinical decision-making or patient care. It is
> not a medical device and makes no regulatory claim.

![viewer](docs/images/viewer.png)

## Contents

- [Why this project](#why-this-project) · [Results](#results) · [Quick start](#quick-start)
- [Dataset setup](#dataset-setup) · [Training](#training) · [Evaluation](#evaluation)
- [Model registry](#model-registry) · [Inference](#inference) · [3D reconstruction and viewer](#3d-reconstruction-and-viewer)
- [Lineage](#lineage) · [Testing](#testing) · [Repository layout](#repository-layout)
- [Limitations](#limitations) · Further reading: [architecture](docs/architecture.md),
  [regulated ML notes](docs/regulated_ml_notes.md), [demo script](docs/interview_demo.md)

## Why this project

Medical imaging ML fails in production less often because of the network and more
often because of everything around it: wrong geometry, silently changed data,
untracked preprocessing, unclear which model produced a result. This repository is
deliberately small (one model, no services, no database) so that the
lifecycle around the model is easy to see and easy to review:

| Question an auditor or engineer asks | Answered by |
|---|---|
| Which scans trained this model, and which were held out? | `artifacts/manifests/<dataset_version>.json` |
| Exactly how were images preprocessed? | `preprocessing_version` (a hash of the config) |
| Which code, seed and config produced the checkpoint? | `artifacts/experiments/EXP-*/experiment.json` |
| What evidence justified releasing it, and who approved it? | registry entry: evaluation + promotion history |
| Which model, preprocessing and input produced *this* prediction? | `python -m src.lineage.trace --inference-id …` |

## Results

RESULTS_PLACEHOLDER

## Quick start

Requires Python ≥ 3.10. A GPU is optional; everything runs on CPU.

```bash
make setup          # venv + CPU PyTorch + package (or see "Windows" below)
make data           # download MSD hippocampus (27 MB, checksum-verified) + manifest
make train-demo     # ~55 min on a laptop CPU → EXP-001
make register       # EXP-001 → model v1.0 (candidate)
make evaluate       # test split → attaches evidence to v1.0
make promote        # candidate → validated (release gate) → production
make infer          # segment a held-out test scan
make mesh           # 3D meshes for that case
make viewer         # interactive 3D viewer
make trace          # lineage of the prediction
make test           # full test suite
```

**Windows (PowerShell)**: every Makefile target is one `python -m …` command:

```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python -m src.data.download --out data
python -m src.data.manifest --config configs/data.yaml
python -m src.training.train --config configs/train.yaml
python -m src.registry.model_registry register --experiment EXP-001 --version v1.0
python -m src.evaluation.evaluate --model-version v1.0 --split test
python -m src.registry.model_registry promote --version v1.0 --to validated --reason "test-split evaluation meets the release gate"
python -m src.registry.model_registry promote --version v1.0 --to production --reason "approved for the demo release"
python -m src.inference.predict --input data/Task04_Hippocampus/imagesTr/hippocampus_017.nii.gz --model-version production
python -m src.reconstruction.mesh --inference-id <INF-ID printed above>
python -m src.visualization.viewer --case <INF-ID>
python -m src.lineage.trace --inference-id <INF-ID>
```

**Docker** (CPU; runs the test suite by default):

```bash
docker build -t medimg3d .
docker run --rm medimg3d
```

## Dataset setup

[Medical Segmentation Decathlon](http://medicaldecathlon.com) **Task04 Hippocampus**:
260 labelled T1-weighted MRI volumes (Vanderbilt University Medical Center, CC-BY-SA 4.0),
two structures: anterior and posterior hippocampus. It is small (27 MB), real, and
trains on a CPU, while still exercising every geometric step a CT/MRI pipeline needs.

```bash
python -m src.data.download --out data      # verifies a pinned SHA-256 before extracting
python -m src.data.manifest --config configs/data.yaml
```

The manifest validates every scan (dimensions, spacing, finite and non-constant
intensities, label/image geometry match, known classes, duplicates), ignores OS
metadata files, and records per scan: `sample_id`, relative `source_path`,
`source_hash`, `shape`, `spacing`, `modality`, `dataset_version`, `split`,
`timestamp`. No patient-identifying information is stored; sample IDs come from
content hashes, not file names.

**No real data?** `python -m src.data.synthetic --out data/synthetic` writes
synthetic NIfTI volumes in the same layout; `configs/smoke/` runs the whole
pipeline on them in seconds (this is what CI does).

**CT data** works through the same code: set `modality: CT` and
`intensity.mode: ct_window` (Hounsfield-unit windowing) in the configs.
**DICOM** series directories are accepted by `src.data.loaders.load_volume`.

## Training

```bash
python -m src.training.train --config configs/train.yaml
```

- Model: compact 3D U-Net in plain PyTorch (`src/models/unet3d.py`, 350k parameters),
  Dice + cross-entropy loss, AdamW, cosine learning-rate schedule.
- Mixed precision when CUDA is available; deterministic algorithms and seeds always.
- Augmentation: left-right flip, intensity scale/shift, Gaussian noise; randomness is derived
  from `(seed, epoch, sample)` so results do not depend on worker scheduling.
- Model selection on the validation split each epoch; `best.pt` / `last.pt` are
  self-describing (architecture, classes, preprocessing, dataset version inside).

The experiment record (`artifacts/experiments/EXP-001/experiment.json`) contains
`experiment_id`, `dataset_version`, `preprocessing_version`, `model_name`,
`model_config`, `training_config`, `random_seed`, `code_version` (git commit +
dirty flag), `best_checkpoint` (+ SHA-256), `metrics`, `timestamp`.

## Evaluation

```bash
python -m src.evaluation.evaluate --model-version v1.0 --split test
```

Runs the **production inference function** on every held-out test scan and scores
the result in the original scan geometry: Dice, IoU, precision, recall per
structure, plus latency. Undefined values (for example precision with no predicted
voxels) are reported as `null` and excluded from means rather than silently
counted. Output: `artifacts/evaluations/EVAL-*.json`, attached to the registry entry.

## Model registry

```bash
python -m src.registry.model_registry register --experiment EXP-001 --version v1.0
python -m src.registry.model_registry list
python -m src.registry.model_registry inspect --version v1.0
python -m src.registry.model_registry promote --version v1.0 --to validated --reason "..."
```

`candidate → validated → production` (and `→ retired`). Promotion to `validated`
requires an evaluation on the split, case count and Dice threshold in
`configs/release.yaml`, on the same checkpoint and dataset version. Every
transition records who, when and why; versions are immutable; checkpoints are
copied into the registry and hash-verified on every load. This demonstrates
controlled model lifecycle management; it is not a claim of regulatory compliance.

## Inference

```bash
python -m src.inference.predict --input scan.nii.gz --model-version production
```

Load → validate against the model's input limits → hash → apply **the model's
registered preprocessing** (the caller cannot override it) → load the hash-verified
checkpoint → segment → keep largest component per structure → map back to the
input's voxel grid → save `segmentation.nii.gz` → write the inference record
(`inference_id`, `input_hash`, `model_version`, `dataset_version`,
`preprocessing_version`, `prediction_hash`, `runtime_ms`, `device`, `timestamp`, ...).

## 3D reconstruction and viewer

```bash
python -m src.reconstruction.mesh --inference-id INF-...
python -m src.visualization.viewer --case INF-...
python -m src.visualization.slices --case INF-...        # static 3-plane overlay PNG
```

Meshes: marching cubes per structure, vertices in scanner millimetre coordinates
(they overlay the scan in any tool), volume-preserving Taubin smoothing, export to
STL, PLY and OBJ. Meshing is independent of the model and checks the mask's hash.

Viewer (PyVista): rotate / zoom / pan; show or hide each structure and the scan
slices; opacity slider; click-click distance measurement in mm; case metadata
(model, dataset, preprocessing, volumes); right panel: scan slices with the
segmentation overlay and a slice slider. `--screenshot file.png` renders off-screen.

## Lineage

```bash
python -m src.lineage.trace --inference-id INF-...
```

Walks prediction → model → evaluation → experiment → dataset → preprocessing →
source scan, and **re-hashes every link** (prediction file, checkpoint, evaluation
file, manifest content, preprocessing config). Exits non-zero if anything was
modified. It also reports whether the input scan was part of the training data.

TRACE_PLACEHOLDER

## Testing

```bash
python -m pytest -q                 # all tests (~1 min on CPU)
python -m pytest -q -m "not slow"   # skip the double-training reproducibility test
```

What is tested: NIfTI and DICOM loading, malformed/invalid volumes, configuration
validation, deterministic preprocessing, resampling, exact inverse mapping of
predictions (including under reorientation), CT windowing, model forward pass,
checkpoint round-trip and tamper detection, bit-identical training reruns, metric
edge cases, registry immutability and release gates, inference output geometry,
rejection of bad input, mesh correctness (watertight, outward, volume within 5 %,
physical coordinates), viewer scene loading, lineage integrity and tamper detection,
and a full end-to-end smoke pipeline on synthetic data. CI (GitHub Actions) runs
lint, format check, mypy and the tests on every push.

## Repository layout

```
configs/            all behaviour (data, preprocessing, experiment, release gate, meshing)
  smoke/            tiny synthetic-data variants used by tests and CI
src/
  config.py         strict, typed config schemas
  data/             loaders (NIfTI, DICOM), validation, manifest, download, synthetic
  preprocessing/    transforms (and their inverse), processed-sample cache
  models/           3D U-Net, checkpoints
  training/         dataset + augmentation, loss, training loop
  evaluation/       metrics, release evaluation
  registry/         model registry and release gate
  inference/        production inference
  reconstruction/   masks → meshes
  visualization/    viewer, slice figures
  lineage/          record layout + verifying trace
  utils/            hashing, logging, reproducibility
tests/              pytest suite
docs/               architecture, regulated ML notes, interview demo
artifacts/          generated (git-ignored): manifests, experiments, registry, inference
```

## Limitations

- Trained on one small public dataset from one site; no subgroup, scanner or
  pathology analysis; performance on other data is unknown.
- Hippocampus crops are small volumes; the pipeline is general, but a whole-body CT
  task would need patch-based training and sliding-window inference.
- The registry and records are local JSON files for a single writer; no access
  control or tamper-evident storage.
- No uncertainty estimation or input-drift detection at inference.
- The viewer is a demonstration, not a clinical viewer (no DICOM display protocols,
  no usability engineering).
- Nothing here has been clinically validated. See the disclaimer above and
  [docs/regulated_ml_notes.md](docs/regulated_ml_notes.md).

## License

Code: MIT. Dataset: MSD Task04 Hippocampus, CC-BY-SA 4.0 (not redistributed here).
