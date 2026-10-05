# Interview demo script (6–8 minutes)

**Before the call:** run the whole pipeline once (README "Quick start"), open
two terminals in the repo with the venv active, close everything else, and have
`artifacts/` populated. Never train live. Set `CASE` to the inference ID you
will show, and run a second inference just before the call so its timestamp is fresh.

```powershell
$CASE = (Get-ChildItem artifacts\inference | Sort-Object LastWriteTime | Select-Object -Last 1).Name
```

---

### 1. Architecture (30 s)

Show the diagram in `README.md`.

> "CT or MRI comes in, is validated and versioned, preprocessed deterministically,
> a PyTorch 3D U-Net segments it, the model goes through a registry with a release
> gate, inference produces a mask and a record, the mask becomes 3D meshes, and
> every output can be traced back. **I intentionally kept the architecture small. In
> medical ML, sophistication is not the same as complexity. I wanted every
> transformation, model and output to be reproducible and traceable.**"

### 2. Data (30 s)

```powershell
python -m src.data.manifest --config configs/data.yaml
```

Open `artifacts/manifests/ds-msd-hippocampus-*.json`, show one record.

> "Sample IDs come from the content hash, never the file name, because file names in
> hospitals often contain patient identifiers. The dataset version is a hash of scans,
> labels and split. The split is hash-based per sample, so when the dataset grows, a
> test scan can never drift into training."

### 3. Training (45 s)

Show `configs/train.yaml`, then `artifacts/experiments/EXP-001/experiment.json`.

> "Everything is config; unknown keys are rejected so a typo cannot silently fall back to
> a default. The record has the git commit, seed, every config, and the checkpoint hash.
> The run is bit-for-bit reproducible: when a run was interrupted and restarted, epoch 1
> had the identical loss to four decimals, and there is a test that asserts identical
> weights across two runs."

### 4. Evaluation (30 s)

Open `artifacts/evaluations/EVAL-*.json`.

> "Evaluation runs the *production* inference function on the held-out test split and
> scores in the original scan geometry, so the numbers include resampling and
> post-processing, not just the network. Dice, IoU, precision, recall per structure,
> and latency."

### 5. Model registry (30 s)

```powershell
python -m src.registry.model_registry list
python -m src.registry.model_registry promote --version v1.0 --to production --reason "skip"
```

(Run the second command on a *candidate* to show the refusal, or describe it.)

> "Promotion is explicit and gated: candidate to validated needs test-split evidence that
> meets `configs/release.yaml`; every transition records who, when and why. One model in
> production; promoting a new one retires the old."

### 6. Inference (45 s)

```powershell
python -m src.inference.predict --input data/Task04_Hippocampus/imagesTr/hippocampus_017.nii.gz --model-version production
```

> "The caller cannot choose preprocessing: the model carries its own registered
> preprocessing, and the checkpoint hash is verified before loading. Output is in the
> input's geometry, plus an inference record."

### 7. 3D visualization (1–2 min)

```powershell
python -m src.reconstruction.mesh --inference-id $CASE
python -m src.visualization.viewer --case $CASE
```

Rotate, zoom, toggle anterior/posterior, lower opacity to see the scan slices through
the structures, scroll the 2D overlay on the right, enable Measure and click two points.

> "Meshes are marching cubes per structure, in scanner millimetre coordinates, so they
> overlay the scan exactly and export to STL/PLY/OBJ for any downstream tool. Meshing is
> separate from the model: it consumes a saved mask and checks its hash. **The 3D viewer
> is not intended to reproduce SpectoMED. It simply demonstrates how I would connect
> medical-image ML output to a usable spatial representation.**"

### 8. Lineage (45 s)

```powershell
python -m src.lineage.trace --inference-id $CASE
```

> "This answers the auditor's questions from records: which scan, which preprocessing,
> which model, which experiment, which data, which metrics justified the release. Every
> link is re-hashed, not just looked up; edit a manifest or a mask and this fails. It also
> tells me this scan was in the test split, never seen in training."

### 9. Testing (30 s)

```powershell
python -m pytest -q
```

> "Tests cover what actually breaks in medical imaging: geometry round-trips under
> reorientation, malformed inputs, metric edge cases, checkpoint tampering, release
> gates, lineage tamper detection, and an end-to-end smoke pipeline that also runs in CI."

### 10. Regulated engineering (45 s)

Open `docs/regulated_ml_notes.md`.

> "This is not compliant with anything, and compliance is mostly process: IEC 62304,
> ISO 14971 risk management, an ISO 13485 QMS. But the engineering decides how
> expensive that process is. If every output is traceable and every change is versioned
> and gated, design-control evidence becomes a by-product instead of an archaeology
> project. **I focused not only on model training but on the entire lifecycle: data,
> preprocessing, training, evaluation, release, inference and traceability.**"

---

## Likely CTO questions and short answers

**Why not nnU-Net / MONAI?** nnU-Net is the right baseline for accuracy and I would
benchmark against it; here the goal was a transparent lifecycle. MONAI is a one-file
swap for the network.

**How would this scale to a team?** Same interfaces, different storage: registry and
records into a database or MLflow with access control; artifacts into object storage
with immutability; training jobs on GPUs; the trace logic stays.

**What about DICOM in production?** The loader has a DICOM-series adapter that reads
voxels and geometry only; tags (PHI) are never copied into records. Real deployments
need series selection, de-identification policies and PACS integration.

**How do you know the model is good enough?** On this dataset, test Dice is reported per
structure; but "good enough" is a clinical question: acceptance criteria, subgroup
analysis, failure-case review with clinicians, and monitoring after release.

**What would you do next?** Subgroup/stratified evaluation, uncertainty estimates per
prediction, input-distribution drift checks at inference, and an append-only record
store.
