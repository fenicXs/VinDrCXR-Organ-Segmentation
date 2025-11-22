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

Notes
- I did not push to your GitHub remote automatically (no auth). Run the commands above locally — Git will prompt for credentials or use your configured SSH key.
- The exported files have full-line comments removed; the code is functionally unchanged.

Attribution
- Original implementation: https://github.com/HuCaoFighting/Swin-Unet
- Dataset and split info from the CheXmask dataset: https://github.com/ngaggion/CheXmask-Database

If you want, I can:
- create a branch and attempt to push (you will need to supply a personal access token or configure SSH access), or
- open a pull request from a generated patch file.
