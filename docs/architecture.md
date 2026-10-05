# Architecture and design choices

## The modules

Each folder in `src/` does one job.

- `config.py`: reads the YAML files in `configs/` and checks them. Every setting must be
  there, and unknown keys are an error.
- `data/`: loads NIfTI files and DICOM folders, checks that a scan is usable, and builds
  the dataset manifest.
- `preprocessing/`: turns a scan into the fixed-size input the network needs, and maps the
  network output back onto the original scan.
- `models/`: the 3D U-Net, and saving/loading checkpoints.
- `training/`: the training loop, augmentation and loss.
- `evaluation/`: metrics, and the evaluation on the test scans.
- `registry/`: model versions and their status (candidate, validated, production, retired).
- `inference/`: segments one new scan and writes a record.
- `reconstruction/`: turns a segmentation into 3D meshes.
- `visualization/`: the 3D viewer and the slice images.
- `lineage/`: where the records are stored, and the trace command.

The flow goes in one direction: data, then preprocessing, training, evaluation, registry,
inference, meshes and viewer. The trace only reads records. It never changes anything.

## Why I built it this way

**Plain PyTorch instead of MONAI.** For one small model I wanted every layer visible in one
short file (`models/unet3d.py`). In a bigger team with many models I would use MONAI. Swapping
it in would only change that one file.

**SimpleITK for all geometry.** Most bugs in medical imaging pipelines come from orientation,
spacing and origin. SimpleITK reads NIfTI and DICOM the same way and resamples in real-world
millimetres, so mapping a prediction back onto the original scan is exact. There is a test
for this: if I pass in the true label as the "prediction", I get the original label back
voxel for voxel, even after reorienting the scan.

**Versions come from content.** The dataset version is a hash of the scans and the split.
The preprocessing version is a hash of its settings. Nobody has to remember to bump a
version number, and two different setups can never end up with the same version.

**The split is fixed per scan.** Whether a scan is train, validation or test depends only on
its ID and a seed. When new scans are added later, the old test scans stay in test. Otherwise
a test scan could slip into training in the next dataset version.

**Evaluation uses the inference code.** The evaluation calls the same `segment()` function
as the inference command and compares in the original scan size. So the test numbers include
everything that happens in real use (resampling, mapping back, post-processing), not just the
network.

**JSON files instead of a database or MLflow.** For one person on one machine, plain files are
easy to read, diff and review, and there is no server to run. If this grew into a team tool,
I would move the registry and records into a database with user accounts. The code that uses
`ModelRegistry` would not need to change.

**Prediction hash from voxels, not from the file.** Compressed NIfTI files contain a timestamp,
so saving the same mask twice gives two different file hashes. Hashing the voxel data gives a
stable value.

**Meshes in scanner coordinates.** The mesh points use the scan's origin, spacing and
direction, so the meshes sit exactly on the scan in any viewer. A test with a sphere checks
that the mesh is closed, faces outward, and has the right volume (within 5%).

## What I left out on purpose

Kubernetes, microservices, a web frontend, a database, cloud storage, an experiment tracking
server, multi-GPU training and model ensembles. None of them would make the main point (a
traceable, repeatable ML workflow) clearer. They would just add more things to run and maintain.
