# UPAR 2027 Challenge, Track 1: pedestrian attribute recognition

Code, data products and instructions to reproduce our two Track 1 systems:

| Entry | Role | Test score | MEVID | Synth | Package md5 |
|---|---|---|---|---|---|
| **X2F-CMASK** | final ranked entry | **0.8017** | 0.7487 | 0.8547 | `e9bb5cdf6d567767d8d0a47791b4d299` |
| **C4** | clean reference | 0.7966 | 0.7471 | 0.8462 | `b45d3f00362f15a1b99f060296a0aaef` |

Score: for each test domain, the harmonic mean of label-based mean accuracy (mA) and instance-F1; the challenge score is the mean over the two domains (MEVID, Synth).

X2F-CMASK ranked 2nd of 89 entries on the final Track 1 leaderboard (Codabench user `nnnnn`).

Team: Trần Trọng Nhân (leader), Nguyễn Chơn Nhân, Nguyễn Thị Ái Trâm, University of Information Technology, Vietnam National University, Ho Chi Minh City, Vietnam. Contact: 24521243@gm.uit.edu.vn.

> **Disclosure.** C4 uses only our validation set for every choice. X2F-CMASK does not:
> 1. it routes images by their position in the test list (positions 0–28,094 are MEVID, 28,095–29,247 are Synth);
> 2. its tail recipe (members, weights, flip TTA) was chosen by comparing leaderboard scores;
> 3. it uses CMASK, which failed our pre-registered validation gate, and was kept after its test score was seen.
>
> No test image or label was used for training or threshold calibration.

## 1. Systems

All members are ViT backbones fine-tuned for the 40 attributes. Members are fused by a weighted average of logits, followed by per-attribute thresholds calibrated on our validation set. Weights are shipped as per-channel int8, and `run.py` has a time guard for the 5-minute A10 limit of the scoring server.

| Member | Backbone (pretrained) | Frozen blocks | Train input | Labels / recipe | Best epoch | Val | Used in |
|---|---|---|---|---|---|---|---|
| F1 | DINOv3 ViT-H+/16 | 8 | 256×128 | LP-FT | 5 | .8944 | C4 |
| L2-soup | DINOv2 ViT-L/14, 4 registers | 0 | 252×126 | LP-FT, uniform soup of seeds 42 and 2 | 7, 7 | .8939, .8952 | C4 |
| F8c | DINOv3 ViT-H+/16 | 8 | 256×128 | CMASK | 5 | .8946 | X2F-CMASK |
| L2c | DINOv2 ViT-L/14, 4 registers | 0 | 252×126 | CMASK | 7 | .8938 | X2F-CMASK |
| G | DINOv3 ViT-H+/16 | 20 | 256×128 | original labels | 7 | .8911 | X2F-CMASK tail |
| R | C-RADIOv4-H | 12 | 256×128 | original labels | 5 | .8978 | X2F-CMASK tail |

Epochs are counted from 0. "Val" is the validation score at threshold 0.5, averaged over Market-1501, PA-100K and PETA.

- **C4:** 0.6·F1 + 0.4·L2-soup at 224×112 on every test image, one threshold set.
- **X2F-CMASK:**
  - head (first 28,095 images): 0.6·F8c + 0.4·L2c at 224×112, thresholds `thresholds_solo`;
  - tail (remaining 1,153 images): 0.24·F8c + 0.16·L2c + 0.3·G + 0.3·R at 192×96 (196×98 for DINOv2), logits averaged with the horizontal flip, thresholds `thresholds` calibrated on the four-member validation logits.

**Pretrained weights**
- DINOv3 ViT-H+/16: `timm/vit_huge_plus_patch16_dinov3.lvd1689m` (Hugging Face, timm).
- DINOv3 ViT-L/16, only for the CMASK out-of-fold models: `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated; needs a Hugging Face token).
- DINOv2 ViT-L/14-reg4: https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
- C-RADIOv4-H: `nvidia/C-RADIOv4-H` (Hugging Face). Uses CLIP normalisation (mean 0.481/0.458/0.408, std 0.269/0.261/0.276).

No external images or labels are used.

## 2. Repository layout

```
notebooks/upar_e2e_08017.ipynb   self-contained pipeline: challenge data -> X2F-CMASK zip
inference/x2f_cmask/             run.py, model.py, metadata.yaml, assets/config.json of the submitted zip
inference/c4/                    same for C4
inference/predict.py             runs a package on a list of images: predictions, labels, challenge metric (Section 9)
training/x2f_cmask/              the scripts/notebooks that produced the X2F-CMASK checkpoints
training/c4/                     the scripts/notebooks that produced the C4 checkpoints
packaging/                       int8 packing, threshold calibration and zip assembly used for both entries
data/split_0710/                 our train/val re-split (image lists)
data/cmask/                      CMASK flags (suspected noisy label cells) on our training set
requirements.txt, Dockerfile
```

- Trained weights are not stored in git (5 GB); see Section 6.
- `inference/*/assets/config.json` differs from the file in the zip only in free-text provenance fields, where Kaggle account names were replaced by `<kaggle-user>`. The same replacement was applied to Kaggle input paths in `training/` and `packaging/`.
- Comments in the original training code are partly in Vietnamese; this README describes everything needed to run it.

## 3. Environment

- Scoring server: one NVIDIA A10 (24 GB), torch 2.4.1, no network, 5 minutes per run.
- Training: Kaggle 2×T4 (16 GB each) for L2, L2c, G, R and the CMASK out-of-fold models; one A100 40 GB (Colab) for F1 and F8c. Freezing only 8 blocks of ViT-H+/16 does not fit on a T4.
- Code: `git clone https://github.com/nguyennhan2006/WACV_UPAR_track1`; weights: Section 6.
- `pip install -r requirements.txt`, or `docker build -t upar-t1 .` (CUDA 12.1, torch 2.4.1). The image runs `inference/predict.py` (Section 9); for training, start it with `--entrypoint bash`.

## 4. Data

1. Get the UPAR 2027 Track 1 annotations from the organisers and the Market-1501, PA-100K and PETA images. The e2e notebook downloads the images from the links given by the organisers.
2. **Re-split.** We pool the official train and validation sets and draw 15% of each source dataset as our validation set, grouped by identity (Market-1501: person ID; PA-100K: blocks of 20 consecutive indices; PETA: blocks of 5) and stratified over the 40 attributes (iterative stratification, seed 0). Result: 111,296 training and 19,780 validation images. `training/x2f_cmask/make_resplit.py` rebuilds it deterministically; `data/split_0710/{train,val}_images.txt` list the images of each part (labels are not redistributed). Every member, fusion weight and threshold uses this split.
3. **CMASK** (X2F-CMASK only). Two DINOv3 ViT-L/16 models are trained out-of-fold on the two halves of our training set (256×128, 8 epochs, last epoch). A cell (image, attribute) is flagged when the out-of-fold logit is more than 2 away from the logit of the calibrated threshold, against the label, and the Platt-scaled probability is ≥ 0.9 for label 0 or ≤ 0.1 for label 1. Positive labels of Backpack, Bag, Glasses-Normal, Glasses-Sun and Hat are never flagged. Flagged cells are removed from the loss; labels are not changed. 19,917 cells in 14,585 images are flagged.
   - `data/cmask/cmask_flags.npz`: `images` (paths as in `gt.csv`), `attr` (40 names), `flag` = `np.packbits(bool[n_images, 40], axis=1)`; md5 `2311665c29d0adad69845cb4dbfbcde4`.
   - `data/cmask/cmask_per_attr.csv`: counts per attribute and dataset.

## 5. Training recipe (all members)

| Item | Value |
|---|---|
| Optimiser | AdamW, backbone LR 3e-5 with layer-wise decay 0.85, head LR 1e-3, weight decay 0.05 |
| Schedule | 8 epochs, 1 warm-up epoch, cosine |
| Batch | 64, mixed precision, gradient clipping 5.0 |
| Regularisation | drop path 0.15, head dropout 0.2, label smoothing 0.05, EMA 0.999 |
| Loss | BCE weighted per attribute by its positive rate (DeepMAR), masked for unlabelled cells (and CMASK cells) |
| Sampler | per-image weight ∝ (size of its source dataset)^−0.5 |
| Head | [CLS ‖ mean patch token] → dropout → Linear(2C, 40); bias initialised to logit(positive rate) |
| Augmentation | low-resolution simulation (p 0.4, width 20–64 px), JPEG (p 0.3, quality 25–90), resize, horizontal flip, pad 8 + random crop, colour jitter 0.25/0.25/0.1 without hue (p 0.8), Gaussian blur (p 0.15), random erasing (p 0.3) |
| Epoch selection | EMA weights of the epoch with the best validation score at threshold 0.5 |
| Seed | 42 (second L2 seed: 2) |

LP-FT (C4): the head is trained first with a frozen backbone, then the whole model is fine-tuned.

Thresholds: per attribute, coordinate descent over 0.05–0.95 in steps of 0.025, two passes, maximising the validation challenge score averaged over the three source datasets. The calibrated thresholds are folded into the logits, so the platform's fixed 0.5 threshold applies them.

## 6. Trained weights

All files are in one Google Drive folder, [UPAR2027_track1_weights](https://drive.google.com/drive/folders/1mEwm1fsJFmdYctWsvFXMlNL6LqswwdCU). It also holds a `README.txt`, the `meta.json` of each checkpoint (next to its `best_ema.pt`), and copies of the backbone licenses in `licenses/`. Read the license terms (Section 10) before using the weights.

| File | md5 | Link |
|---|---|---|
| X2F-CMASK package (zip, 2.64 GB) | `e9bb5cdf6d567767d8d0a47791b4d299` | [Drive](https://drive.google.com/file/d/1YrnN5wIJYCr0r_SD9Fqmz6uxAB5pv39H/view?usp=sharing) |
| C4 package (zip, 1.15 GB) | `b45d3f00362f15a1b99f060296a0aaef` | [Drive](https://drive.google.com/file/d/19uzMts7ws6smvgca-BxKuI2zEzebwMF5/view?usp=sharing) |
| F8c `best_ema.pt` | `6960557c3b5b09972ba4aa38493866d6` | [Drive](https://drive.google.com/file/d/1IbyMJ8PRAQ1XQ6wzGcGGtEbFAFwnf27R/view?usp=sharing) |
| L2c `best_ema.pt` | `5f20fe0e2f4dd7ddb4ce952186e47d91` | [Drive](https://drive.google.com/file/d/1uiew9qCJftHgeXfKg4h7rwYcSy7-WlTZ/view?usp=sharing) |
| G `best_ema.pt` | `d3d0513a27101f0337698c8f3d184fd6` | [Drive](https://drive.google.com/file/d/1268TURkk_PN2QTJot-XSF24bCn9AZLbu/view?usp=sharing) |
| R `best_ema.pt` | `945dfe3617d88a066df72c04fb49fc60` | [Drive](https://drive.google.com/file/d/12o43do3Fk0moAnWJJzDH0xrSjx0caskN/view?usp=sharing) |

From the command line: `pip install gdown==5.2.0`, then `gdown --folder 1mEwm1fsJFmdYctWsvFXMlNL6LqswwdCU` for the whole folder, or `gdown <file id>` for one file (the id is the part after `/d/` in the link). Check the md5 after downloading.

F1 and L2-soup (uniform weight average of L2 seeds 42 and 2) are released only inside the C4 package, as `assets/weights_0.pt` (F1, w = 0.6) and `assets/weights_1.pt` (L2-soup, w = 0.4), in the int8 form that was submitted. Their full-precision checkpoints are not released.

Each `best_ema.pt` is the `state_dict` of `PARNet` in `inference/*/model.py`:

```python
import json, torch
from model import PARNet
meta = json.load(open("meta.json"))
m = PARNet(meta["arch"], pretrained=False, img_h=meta["img_h"], img_w=meta["img_w"],
           parts=meta["parts"], qdec=meta["qdec"], cmid=meta["cmid"], chist=meta["chist"])
m.load_state_dict({k: v.float() for k, v in torch.load("best_ema.pt", map_location="cpu").items()})
```

Inputs are normalised with `mean`/`std` from the checkpoint's `meta.json`; outputs are 40 logits in the column order of `gt.csv`.

## 7. Reproducing X2F-CMASK

`notebooks/upar_e2e_08017.ipynb` contains the whole pipeline: re-split, two out-of-fold models, CMASK flags, the four members, int8 packing, threshold calibration, zip assembly and self-checks. It imports no local file; `run.py` and `model.py` are written by `%%module` cells and checked byte for byte (CRC) against the submitted versions.

It takes about 30 GPU-hours, so it runs as several Kaggle sessions selected by the environment variable `UPAR_STAGES`. Each session mounts the outputs of the previous ones as inputs.

| Session | Hardware | `UPAR_STAGES` | Time | Needs |
|---|---|---|---|---|
| A | 2×T4 | `oof0,oof1,flags` | ~6 h | — |
| B | 2×T4 | `l2c` | ~5 h | A |
| C | 2×T4 | `g11` | ~7 h | — |
| D | 2×T4 | `radio` | ~6.5 h | — |
| E | GPU ≥ 40 GB | `f8c` | ~2 h | A |
| F | any GPU | `pack` | ~30 min | B, C, D, E |

- A training run that exceeds `UPAR_TIME_LIMIT_H` (default 11 h) saves `resume.pt` and continues in the next session.
- The DINOv3 ViT-L/16 download needs a Hugging Face token (`HF_TOKEN`, or the Kaggle secret named in the notebook). The token is never printed or written.
- **Packing only.** Put the four released checkpoints in folders `e2e_f8c`, `e2e_l2c`, `e2e_g11`, `e2e_radio` (each with `best_ema.pt` and `meta.json`), mount them as input and run `UPAR_STAGES=pack`.
- **Outside Kaggle (Docker).** Put the challenge data and the outputs of earlier sessions in one folder, laid out as the Kaggle inputs of that session, then
  ```bash
  docker run --rm -it --gpus all --ipc=host --entrypoint bash -v /path/to/inputs:/inputs upar-t1
  # inside the container:
  UPAR_INPUT_ROOT=/inputs UPAR_STAGES=oof0,oof1,flags \
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 notebooks/upar_e2e_08017.ipynb
  ```
  Outside Kaggle the outputs go to `notebooks/work/`; mount a host folder there (`-v "$PWD/work:/workspace/upar-t1/notebooks/work"`) to keep them.
- GPU training is not bit-exact, so retrained weights, CMASK flags and thresholds differ slightly; the deterministic parts (split, `run.py`, `model.py`, `metadata.yaml`, config structure) are checked by md5/CRC. The notebook has passed static checks (compile, pyflakes, CRC); a full end-to-end run has not been done. The submitted checkpoints were produced by the original code in `training/x2f_cmask/`.

## 8. Reproducing C4

1. Split: as in Section 4.
2. F1: `training/c4/a100_F1_f8_lpft.py` on one A100 40 GB (about 12.6 min per epoch).
3. L2, two seeds: `training/c4/upar_gpu_dinov2_vitl14_reg_lpft_resplit.ipynb` (seed 42) and `..._lpft_s2_resplit.ipynb` (seed 2), each on Kaggle 2×T4 (about 35 min per epoch).
4. Package: `packaging/upar_pack_c4_1017_src.py` (Kaggle cell script). It averages the two L2 EMA checkpoints uniformly (soup), then calls `packaging/pack_hplus.py` in `pair` mode, which
   - converts the weights to per-channel int8,
   - interpolates the DINOv2 position embedding from 18×9 to 16×8 patches (224×112),
   - computes the fused validation logits with the packaged `run.py` (`packaging/run_v17x.py`) and calibrates the thresholds (helpers from `packaging/upar_v2_src.py`),
   - writes the zip. An earlier package mounted as `--a` supplies `model.py` and the config template.

X2F-CMASK was assembled the same way: `pack_hplus.py` for the F8c + L2c pair, then `packaging/assemble_split_1019.py`, which adds G and R, the split configuration and `run_split.py` as `run.py`.

## 9. Inference

### Reproducing the leaderboard predictions

`inference/predict.py` runs a submitted package on a list of images with the package's own `run.py`, through the same calls as the Codabench scoring program. With Docker (needs the NVIDIA Container Toolkit; the leaderboard runs used one A10 with 24 GB):

```bash
git clone https://github.com/nguyennhan2006/WACV_UPAR_track1
cd WACV_UPAR_track1
docker build -t upar-t1 .
docker run --rm --gpus all \
  -v /path/to/test/images:/data:ro \
  -v "$PWD/packages:/workspace/upar-t1/packages" \
  -v "$PWD/out:/out" \
  upar-t1 --package x2f_cmask --images /data --out /out/predictions.csv
```

Without Docker: `pip install -r requirements.txt`, then `python inference/predict.py` with the same arguments.

- `--package x2f_cmask` (the final entry) or `--package c4` downloads the submitted zip once from Google Drive into `packages/`, checks its md5 and unpacks it. A zip downloaded by hand (Section 6) or an unpacked package folder can be given instead.
- `--images` is a folder of images, or a CSV whose first column is the image path relative to `--root` (for example `MEVID/images/0.jpg`; a header such as `# image,Age-Young,...` is optional). If the CSV also has the 40 label columns, the script prints the challenge metric: for each top-level folder (domain) the harmonic mean of mA and instance F1, then the mean over domains.
- Outputs: `predictions.csv` (40 probabilities per image; the calibrated thresholds are folded in, so 0.5 is the decision threshold), `predictions_labels.csv` (0/1) and, when labels are given, `predictions_scores.json`.
- `--limit 256` runs only the first 256 images, as a quick check.

Two settings make the result independent of the machine:

- **Routing.** X2F-CMASK sends MEVID images to the F8c + L2c pair and synthetic images to the four-model tail. On Codabench the test stream was MEVID (positions 0–28,094) followed by Synth (28,095–29,247), and `run.py` split it by position. `predict.py` feeds the images under a top-level folder named `Synth*` last and puts the split there, so the routing is the same for any list order. `--split-at N` routes by position instead.
- **Time guard.** To stay under the 5-minute limit, `run.py` lowers the input resolution or drops the tail models when it projects that a run will be too slow. `predict.py` switches this off by default (`--time-guard on` keeps it), so every image runs as on the A10: X2F-CMASK runs MEVID at 224×112 and Synth at 192×96 (196×98 for DINOv2) with all four models and horizontal-flip averaging; C4 runs both models on every image at 224×112. Do not use `UPAR_GUARD_OFF=1` for this: it also removes the fixed lower-resolution tier of the tail.

GPU kernels are not bit-exact across hardware, so a few scores very close to 0.5 may flip.

### Python API

The scoring program imports `run.py` from the package root, calls `load_model()` once, then `predict_batch(samples)` (or `predict_image(sample)`). Each sample is a dict with the image file under `image_path`; on Codabench it also carried `image` (the relative path), `index` (the position in the stream) and `attribute_names` (the order of the 40 outputs). `run.py` returns 40 scores per image, and the platform thresholds them at 0.5.

```python
import sys; sys.path.insert(0, "packages/x2f_cmask")   # the folder that contains run.py
import run
run.load_model()
probs = run.predict_batch([{"image_path": "/data/MEVID/images/0.jpg", "image": "MEVID/images/0.jpg", "index": 0}])
```

Called this way, `run.py` counts images in the order they arrive, so the X2F-CMASK routing matches the challenge only when the MEVID images come first (or `UPAR_SPLIT_AT` is set to their count).

## 10. License and citation

| Part | License |
|---|---|
| Code (`inference/`, `training/`, `packaging/`, `notebooks/`, Dockerfile) | MIT, see [LICENSE](LICENSE) |
| Data products (`data/split_0710/`, `data/cmask/`) | CC BY-NC-SA 3.0, because they are derived from the UPAR annotations, see [data/LICENSE.md](data/LICENSE.md) |
| Trained weights and packages (Section 6) | Not covered by the MIT license. Each checkpoint follows the license of its pretrained backbone, and all were trained on non-commercial data: use them for non-commercial research only. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) |

The repository contains no images and no pretrained weights; they are downloaded from their original sources.

If you use this code, please cite this repository ([CITATION.cff](CITATION.cff)) and the UPAR dataset and challenge papers:

```bibtex
@inproceedings{specker2023upar,
  title     = {{UPAR}: Unified Pedestrian Attribute Recognition and Person Retrieval},
  author    = {Specker, Andreas and Cormier, Mickael and Beyerer, J{\"u}rgen},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages     = {981--990},
  year      = {2023}
}
@inproceedings{cormier2024upar,
  title     = {{UPAR} Challenge 2024: Pedestrian Attribute Recognition and Attribute-Based Person Retrieval -- Dataset, Design, and Results},
  author    = {Cormier, Mickael and Specker, Andreas and Junior, Julio C. S. Jacques and Moritz, Lennart and Metzler, J{\"u}rgen and Moeslund, Thomas B. and Nasrollahi, Kamal and Escalera, Sergio and Beyerer, J{\"u}rgen},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  pages     = {359--367},
  year      = {2024}
}
```
