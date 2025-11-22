# VinDr-CXR Organ Segmentation — Swin-UNet

This folder contains a cleaned, export-ready subset of the Swin-UNet segmentation pipeline adapted for the VinDr-CXR organ segmentation split.

Contents
- `SwinSegmentation/` — cleaned Python files from the original SwinSegmentation folder (most full-line comments removed; docstrings preserved).
- `VinDrCXR Official split/` — the two official split text files used in the pipeline.

Why this export
- Prepares a minimal, readable copy of the implementation to push to a new GitHub repo.
- Removes verbose comments while keeping docstrings and important metadata.

Quick start
1. Copy this folder into your local clone of `https://github.com/fenicXs/VinDrCXR-Organ-Segmentation` or push as a new branch.

Commands (PowerShell)
# Initialize a branch and push to your remote (replace <branch-name> as needed)

```powershell
cd "c:\Users\redbl\OneDrive\Documents\Image Processing and analysis\Groups\segmentation\VinDrCXR-Organ-Segmentation"
# Option A: add to existing repo remote (recommended to push on a branch first)
git checkout -b add-swinsegmentation-export
git add export_for_push/SwinSegmentation export_for_push/"VinDrCXR Official split" 
git commit -m "Add cleaned SwinSegmentation export and VinDrCXR split files"
# Ensure your remote is set (replace origin if needed)
# If you haven't added your GitHub repo as a remote yet:
# git remote add origin https://github.com/fenicXs/VinDrCXR-Organ-Segmentation.git
git push origin add-swinsegmentation-export

# Option B: directly copy files into target repo folder then commit & push
# (if this workspace is the repo you want to push to, skip remote add)
```

## Notes on Data Usage

- The CheXmask-derived masks remain governed by the CheXmask-Database license. Ensure compliance with the usage restrictions of each underlying dataset (MIMIC-CXR, CheXpert, VinDr-CXR).
- No raw images are distributed in this repository. Only official metadata tables and curated splits are included.

## Citation

If you use the CheXmask dataset assets included or referenced by this repository, please cite the original release:

```
@misc{gaggion2023chexmaskPhysioNet,
   author = {Gaggion, N. and Mosquera, C. and Aineseder, M. and Mansilla, L. and Milone, D. and Ferrante, E.},
   title = {{CheXmask Database: a large-scale dataset of anatomical segmentation masks for chest x-ray images (version 0.1)}},
   year = {2023},
   howpublished = {PhysioNet},
   note = {https://doi.org/10.13026/dx54-8351}
}
```

For experiments or derivatives relying on the Swin-Unet implementation, please cite the corresponding publications:

```
@InProceedings{swinunet,
   author = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
   title = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
   booktitle = {Proceedings of the European Conference on Computer Vision Workshops (ECCVW)},
   year = {2022}
}

@misc{cao2021swinunet,
   title = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
   author = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
   year = {2021},
   eprint = {2105.05537},
   archivePrefix = {arXiv},
   primaryClass = {eess.IV}
}
```

## Acknowledgements

This work stands on the shoulders of the CheXmask-Database and Swin-Unet teams. Please cite their publications when publishing results obtained from this workflow.
