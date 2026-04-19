# YOLO Object Detection — Project Notes

## Overview

A from-scratch YOLOv2-style object detector trained on COCO 2017. The model takes a 416×416 RGB image and outputs bounding boxes with class labels for up to 80 categories. Everything — architecture, data loading, loss, training loop, and inference — is implemented in PyTorch without using a pre-existing YOLO library.

---

## Files

| File | Purpose |
|---|---|
| `model.ipynb` | Defines the `Conv` block, architecture list, and `Model` class |
| `data.ipynb` | Loads COCO JSON, builds the `coco_dataset` PyTorch Dataset |
| `dataset.py` | Same dataset class as a `.py` file so DataLoader workers can import it |
| `train.ipynb` | Loss function (`total_loss`) and the training loop |
| `main.ipynb` | Inference: loads `model1.pth`, decodes predictions, runs NMS, plots results |
| `model1.pth` | Saved model weights (written after every epoch) |
| `train2017/` | ~118k COCO training images |
| `val2017/` | ~5k COCO validation images (held out — never seen during training) |
| `annotations/` | COCO JSON annotation files |

---

## Architecture (`model.ipynb`)

### Conv Block

Every backbone layer is a `Conv` block — three operations in sequence:

1. **Conv2d** — learns spatial features via a sliding kernel
2. **BatchNorm2d** — normalises activations for stable training
3. **SiLU** — smooth activation function (replaces ReLU in modern nets)

Padding is `kernel_size // 2` to preserve spatial resolution.

### Backbone + Head

The model follows the YOLOv2 design: progressively downsample the image with convolutions and max-pooling to extract rich features, then attach a detection head.

| Stage | Resolution | Key layers |
|---|---|---|
| Input | 416 × 416 | RGB image |
| Stage 1 | 208 × 208 | Conv(32) → MaxPool |
| Stage 2 | 104 × 104 | Conv(64) → MaxPool |
| Stage 3 | 52 × 52 | Conv(128) → Conv(64) → Conv(128) → MaxPool |
| Stage 4 | 26 × 26 | Conv(256) → Conv(128) → Conv(256) → MaxPool |
| Stage 5 | 13 × 13 | Conv(512) → 2× bottleneck → MaxPool |
| Deep features | 13 × 13 | Conv(1024) × 3, Conv(512) × 2 |
| Head | 13 × 13 | Conv(1024) × 2 → raw Conv2d(170) |

**Total layers:** 26 (indices 0–25).

### Output Shape

The final layer outputs `[170, 13, 13]`:
- **13×13** grid cells
- **2 anchor boxes** per cell
- **85 values per anchor** = 1 objectness + 2 (x,y) + 2 (w,h) + 80 class scores

So: 2 × 85 = 170 channels.

### Output Head Design

The last layer (index 25) is a **plain `Conv2d` with no BatchNorm or SiLU**. This is essential: `BCEWithLogitsLoss` expects raw logits. If BN/SiLU were applied, the activations would be pinned near zero and the model could never produce strong "no object" predictions for background cells, causing training to stall.

### Objectness Bias Initialisation

At init, the objectness channels (indices 0 and 85) have their biases set to **−5.8**. This corresponds to `P(object) ≈ σ(−5.8) ≈ 0.003 ≈ 1/338`. Since each image has ~5 objects spread across 2×13×13 = 338 cells, most cells are background. Starting from this prior saves the first few epochs that would otherwise be wasted driving objectness toward a sensible baseline.

---

## Data (`data.ipynb` / `dataset.py`)

### COCO Format

The COCO annotation JSON provides:
- `images` — list of `{id, file_name, width, height}`
- `annotations` — list of `{image_id, category_id, bbox, iscrowd}`
- `categories` — 80 class names (IDs 1–90 with gaps)

Each `bbox` is `[x_topleft, y_topleft, width, height]` in pixels. Crowd annotations (`iscrowd=1`) are skipped — they're imprecise polygons over dense groups.

### Dataset Class

`coco_dataset.__getitem__` returns:
- **Image tensor:** `[3, 416, 416]`, float32, normalised to `[0, 1]`
- **Labels tensor:** `[N, 5]` — one row per object: `[class_idx, cx, cy, w, h]` all in `[0, 1]` relative to image size

Category IDs are remapped from the sparse 1–90 range to clean 0–79 indices via a `cat_to_idx` dict.

### DataLoader

```
batch_size  = 16
num_workers = 4
shuffle     = True (train) / False (val)
```

A custom `yolo_collate` function stacks images into `[B, 3, 416, 416]` but keeps labels as a list of tensors (variable number of objects per image, so they can't be naively stacked).

---

## Loss Function (`train.ipynb`)

### Structure

The `[170, 13, 13]` prediction is split into four tensors:

| Tensor | Shape | Content |
|---|---|---|
| `pred_obj` | `[2, 13, 13]` | Objectness logit per anchor per cell |
| `pred_xy` | `[2, 2, 13, 13]` | (x, y) offset within cell |
| `pred_wh` | `[2, 2, 13, 13]` | (w, h) raw values |
| `pred_cls` | `[2, 80, 13, 13]` | Class logits |

Ground-truth labels are mapped into matching target tensors. For each object:
1. Compute which grid cell contains the centre: `grid_x = int(cx * 13)`, `grid_y = int(cy * 13)`
2. Pick the best-matching anchor by IoU
3. Set `obj_mask[anchor, grid_y, grid_x] = 1`
4. Fill in `target_obj`, `target_xy`, `target_wh`, `target_cls` at that cell

### Loss Components

| Term | Formula | Applied to |
|---|---|---|
| `loss_obj` | BCE (logits vs 1) | Positive cells only |
| `loss_noobj` | BCE (logits vs 0) | Background cells (λ=0.5) |
| `loss_xy` | BCE (predicted offset vs target offset) | Positive cells |
| `loss_wh` | MSE (pred vs log(target/anchor)) | Positive cells |
| `loss_cls` | BCE (class logits vs one-hot) | Positive cells |

**Total:** `loss = λ_obj·loss_obj + λ_noobj·loss_noobj + λ_xy·loss_xy + λ_wh·loss_wh + λ_cls·loss_cls`

**λ values:** `λ_obj = λ_xy = λ_wh = λ_cls = 1.0`, `λ_noobj = 0.5`

### Per-Cell Normalisation (Key Fix)

Each loss term is normalised by dividing by the number of cells that actually contribute — `num_obj` for positive-only terms, `num_noobj` for background. The naive `.mean()` over the full 2×13×13 grid dilutes each positive-cell signal by ~300× (only ~5 of 338 cells have objects), causing training to collapse into a "predict no object everywhere" local minimum.

```python
num_obj   = obj_mask.sum().clamp(min=1)
num_noobj = noobj_mask.sum().clamp(min=1)

loss_obj   = (bce(pred_obj, target_obj)   * obj_mask).sum()              / num_obj
loss_noobj = (bce(pred_obj, target_obj)   * noobj_mask).sum()            / num_noobj
loss_xy    = (bce(pred_xy, target_xy)     * obj_mask.unsqueeze(1)).sum() / num_obj
loss_wh    = (mse(pred_wh, log_targets)   * obj_mask.unsqueeze(1)).sum() / num_obj
loss_cls   = (bce(pred_cls, target_cls)   * obj_mask.unsqueeze(1)).sum() / num_obj
```

---

## Training Loop (`train.ipynb`)

- **Optimiser:** Adam, `lr=0.001`
- **Device:** Apple Silicon MPS (Metal Performance Shaders)
- **Training set:** ~118k images (train2017), ~7,400 steps per epoch at batch=16
- **Validation:** Full val2017 (~5k images) after every epoch under `model.eval()` + `torch.no_grad()`
- **Checkpoint:** `model1.pth` saved after every epoch

Each training step: zero grad → forward pass (batch) → compute loss per image → average over batch → backward → step.

---

## Inference (`main.ipynb`)

The `predict(image_path)` function:
1. Loads and resizes the image to 416×416
2. Runs the model → `[170, 13, 13]` logits
3. For each anchor in each cell: applies sigmoid to objectness/xy/class, `exp` to wh
4. Keeps boxes above `CONF_THRESH = 0.9`
5. Decodes grid-relative coordinates back to pixel space
6. Runs **NMS** (`NMS_THRESH = 0.4`) to remove duplicate boxes around the same object
7. Returns boxes with class name and confidence

A random image is picked from `val2017/` on each run so you're not always testing the same image.

---

## Training Results & Overfitting

### Early Epochs (0–8): Healthy Learning

Training started well. Both train and val loss fell steadily together, with all components improving:

```
Epoch 0  | train 6.83 | val 6.26
Epoch 1  | train 5.75 | val 5.42
Epoch 2  | train 5.21 | val 4.97
Epoch 3  | train 4.82 | val 4.69
Epoch 4  | train 4.52 | val 4.48
Epoch 5  | train 4.27 | val 4.42
Epoch 6  | train 4.06 | val 4.31
Epoch 7  | train 3.85 | val 4.23
Epoch 8  | train 3.64 | val 4.21  ← val bottoms out here
```

At epoch 8, val loss hit its minimum of **4.2058**. The loss breakdown at this point:

| Component | Value | What it means |
|---|---|---|
| obj | 0.18 | Model is ~86% confident where objects are |
| noobj | 0.41 | Background suppression still improving |
| xy | 1.28 | Near the BCE entropy floor (~1.0) for continuous targets |
| wh | 0.43 | Width/height prediction learning well |
| cls | 2.11 | Classification is the dominant remaining challenge |

### Epoch 9 Onward: Overfitting Sets In

After epoch 8, val loss started climbing while train loss kept falling — the classic overfitting signature:

```
Epoch 9  | train 3.44 | val 4.25  ← val starts rising
Epoch 10 | train 3.24 | val 4.31
Epoch 11 | train 3.06 | val 4.37
Epoch 12 | train 2.89 | val 4.53
Epoch 13 | train 2.73 | val 4.61
Epoch 14 | train 2.59 | val 4.82
Epoch 15 | train 2.46 | val 5.02
Epoch 16 | train 2.35 | val 5.20
Epoch 17 | train 2.25 | val 5.40
Epoch 18 | train 2.17 | val 5.45
Epoch 19 | train 2.10 | val 5.75
Epoch 20 | train 2.03 | val 5.90
Epoch 21 | train 1.97 | val 5.97  ← training stopped (KeyboardInterrupt)
```

By epoch 21, the gap was **train 1.97 vs val 5.97** — the model was 3× worse on unseen images. It had memorised the training set rather than learning to generalise.

**Which component drove the overfitting?** The `cls` (classification) term was the main culprit — it improved on train but ballooned on val from 2.11 (epoch 8) back up to 3.77 (epoch 21). The model was memorising which class went with which specific training image rather than learning robust visual features. `obj` loss on val also crept up, indicating the model was becoming overconfident about object locations it had seen in training but that didn't generalise.

### Why It Happened

1. **No data augmentation.** Every training image was seen the same way each epoch — no flipping, colour jitter, crop, or mosaic. Augmentation is the single biggest lever against overfitting in object detection.
2. **No weight decay.** Adam with no `weight_decay` imposes no penalty on large weights, allowing the model to sharpen training-set-specific representations unchecked.
3. **No early stopping.** Training continued well past the best checkpoint (epoch 8). The saved `model1.pth` reflects epoch 21, not epoch 8.
4. **Model capacity.** ~30M parameters with up to 1024-channel feature maps has more than enough capacity to memorise 118k images without generalising.

### The Best Checkpoint

The weights from **epoch 8** represent the best generalisation achieved (val 4.21). Everything after that is overfitting. If you still have the epoch 8 checkpoint, use that for inference.

### What Would Help Next

| Fix | Expected impact |
|---|---|
| Random horizontal flip + colour jitter | High — cheapest augmentation, biggest regularisation gain |
| Mosaic augmentation (YOLOv4 style) | High — forces the model to detect objects at unusual scales/crops |
| Weight decay (`weight_decay=1e-4` in Adam) | Medium — adds L2 regularisation without changing anything else |
| Early stopping (stop when val hasn't improved for N epochs) | Preserves the best checkpoint automatically |
| LR cosine decay | Medium — helps the model settle rather than oscillate in the loss landscape |
| Dropout before the head | Low-medium — adds noise to prevent overconfident representations |
