# Architecture and design decisions

## Data flow

```
                 configs/*.yaml  (all behaviour; validated, no hidden defaults)
                        │
  NIfTI / DICOM ──▶ data.loaders ──▶ data.validation ──▶ data.manifest
                                                           │  dataset_version = hash(scans, labels, split)
                                                           ▼
                                         preprocessing.transforms (+ cache)
                                           │  preprocessing_version = hash(config)
                                           ▼
                                   training.train (PyTorch 3D U-Net)
                                           │  experiment record: git commit, seed, configs, ckpt hash
                                           ▼
                           registry.model_registry  ◀── evaluation.evaluate (test split)
                           candidate → validated → production   (release gate)
                                           │
  new scan ──▶ inference.predict ──▶ segmentation.nii.gz + inference record
                                           │
                       reconstruction.mesh ──▶ STL / PLY / OBJ per structure
                                           │
                visualization.viewer / slices        lineage.trace (verifies every link)
```

## Module responsibilities

| Module | Responsibility | Depends on |
|---|---|---|
| `config.py` | Typed, strict configuration schemas | – |
| `data/loaders.py` | Read NIfTI and DICOM into one `Volume` type; hash inputs | SimpleITK |
| `data/validation.py` | Reject volumes the pipeline was not built for | loaders |
| `data/manifest.py` | Versioned dataset listing and deterministic split | loaders, validation |
| `preprocessing/transforms.py` | Reorient, resample, normalise, crop/pad, and the inverse | SimpleITK |
| `preprocessing/cache.py` | Materialise processed samples keyed by both versions | transforms |
| `models/unet3d.py` | The network (plain PyTorch) | torch |
| `models/checkpoint.py` | Self-describing, hash-verified checkpoints | unet3d |
| `training/` | Dataset/augmentation, loss, training loop, experiment record | all of the above |
| `evaluation/` | Metrics and the release evaluation | inference |
| `registry/` | Model lifecycle and release gate | records |
| `inference/predict.py` | Production inference + inference record | registry, preprocessing |
| `reconstruction/mesh.py` | Mask → surface meshes (independent of the model) | scikit-image, trimesh |
| `visualization/` | Viewer and static figures | pyvista, matplotlib |
| `lineage/` | Record layout/IO and the verifying trace | everything (read-only) |

## Design decisions

**Plain PyTorch U-Net instead of MONAI.** MONAI is excellent, and in a
product team with many architectures I would use it. For one small model, a
~80-line network is easier to review and test, and leaves no unseen defaults
between the config and the computation. Swapping in `monai.networks.nets.UNet`
would change one file.

**SimpleITK for all image geometry.** Orientation, spacing, origin and
direction are where medical-imaging pipelines silently go wrong. SimpleITK
handles NIfTI and DICOM through one API and resamples in physical space, which
makes the inverse mapping (prediction → original scan grid) exact. This is
tested: a "perfect prediction" restores to the original label voxel-for-voxel,
including under reorientation.

**Content-derived versions.** `dataset_version` and `preprocessing_version`
are hashes of what defines them. Nobody has to remember to bump a number, and
two different things can never share a version.

**Hash-based split.** Each sample's split is a function of `(seed, sample_id)`.
When the dataset grows, existing test scans stay in test, which prevents a
subtle form of leakage between dataset versions.

**Evaluation through the production path.** The evaluator calls the same
`segment()` function as the inference CLI and scores in the original scan
geometry. Metrics therefore include the effects of resampling, inverse mapping
and post-processing, not just the network.

**JSON files, not a database or MLflow.** For one team on one machine,
plain files are reviewable with `git diff`, `jq` and a text editor, and
have no service to run. The registry is designed for a single writer; the
upgrade path (a database with access control, or MLflow's registry) is a
storage change behind the same `ModelRegistry` interface.

**Prediction hash over voxels, not the file.** Compressed NIfTI files embed a
gzip timestamp, so the same mask saved twice has two file hashes. The voxel
array hash is stable and means what we care about.

**Meshes in scanner coordinates.** Vertices are mapped through the scan's
origin, spacing and direction matrix, so meshes overlay the scan exactly in
any tool that respects DICOM/ITK physical space. Triangle winding is
corrected when the direction matrix contains a reflection (tested with a
sphere: watertight, outward-facing, volume within 5 %).

## What was deliberately not built

Kubernetes, microservices, a web frontend framework, a database, cloud
storage, an experiment-tracking server, multi-GPU training, model ensembling,
and any diagnostic claim. Each would add operational weight without making
the central point (a traceable, reproducible ML lifecycle) clearer.
