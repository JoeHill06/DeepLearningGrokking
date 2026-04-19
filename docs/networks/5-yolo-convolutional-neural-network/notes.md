# COCO YOLO Convolutional Neural Network (PyTorch)

A YOLOv2-style object detection network built from scratch with PyTorch, trained on the MS COCO 2017 dataset. The model divides the image into a 13×13 grid and predicts 2 bounding boxes per cell with class probabilities across 80 categories — 170 output channels total. Unlike the previous Keras attempt, this model successfully learned to detect objects, but eventually overfit the training set.

### Comments
This model was to far out of reach and required me to use help form tools like claude. 

## Architecture decisions

- **26-layer backbone** following YOLOv2 — interleaved Conv+BN+SiLU blocks and MaxPool layers downsample from 416×416 to 13×13 over 5 stages
- **Bottleneck pairs** (e.g. Conv(256)→Conv(128)→Conv(256)) at 52×52 and 26×26 — compress and expand channels to learn richer features without exploding parameter count
- **Plain Conv2d output head** with no BN or activation — required for `BCEWithLogitsLoss` to function correctly
- **2 anchors per cell** (`[0.28, 0.22]` and `[0.38, 0.48]`) — pre-set box priors matched to typical COCO object aspect ratios, reducing what the model needs to learn
- **Objectness bias init at −5.8** — prior matching `P(object) ≈ 1/338` per cell
- **Per-cell loss normalisation** — each loss term divided by the number of positive (or negative) cells, not the full grid size
- **Adam lr=0.001** — 10× higher than the previous attempt; the corrected loss scale made this stable

## What I learned

- How to fix a broken loss function — the original `.mean()` over the full 2×13×13 grid diluted each positive-cell signal by ~300× (only ~5 of 338 cells contain objects per image), causing the model to collapse into predicting no object everywhere. The fix was to normalise each loss term by the number of cells that actually contribute
- How to initialise the objectness bias to match the data prior — setting the objectness bias to −5.8 gives `P(object) ≈ 1/338`, saving the first few epochs from being wasted driving objectness to a sensible baseline
- What overfitting looks like in the training log — train loss kept falling (6.83 → 1.97) while val loss bottomed at epoch 8 (4.21) and then rose back to 5.97 by epoch 21
- How to read a YOLO loss breakdown — classification (`cls`) was the dominant term throughout and was the first to diverge between train and val, showing the model was memorising class-appearance patterns rather than generalising
- Why data augmentation matters  — the model had ~30M parameters and no augmentation, so it could memorise 118k training images without learning to generalise to unseen ones


## Why it overfit

The model learned successfully for 8 epochs then overfit heavily. By epoch 21: train 1.97 vs val 5.97 — a 3× gap. The classification term drove the divergence, rising from 2.11 (epoch 8) to 3.77 (epoch 21) on val while continuing to fall on train. The causes were straightforward: no data augmentation (every image seen identically each epoch), no weight decay (no penalty on large weights), and no early stopping (training ran past the best checkpoint). With ~30M parameters and 118k training images seen repeatedly without variation, the model had capacity to memorise rather than generalise.

## Results

- Training ran for 22 epochs before being stopped
- Best val loss: **4.21** at epoch 8
- Final train loss: 1.97 — Final val loss: 5.97 (epoch 21) almost 48 hours
- Best checkpoint (`model1.pth`) corresponds to epoch 8 — all later epochs are overfit
- The model did learn real detections — objectness confidence at epoch 8 was ~86% on object cells, classification and localisation both improved meaningfully before overfitting set in.


