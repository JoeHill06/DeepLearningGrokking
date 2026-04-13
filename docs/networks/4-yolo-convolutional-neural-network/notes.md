# COCO YOLO Convolutional Neural Network (Keras)

A YOLO-style object detection network built from scratch with TensorFlow/Keras, trained on the MS COCO 2017 dataset. Rather than classifying a whole image, this network divides the image into a 13×13 grid and predicts bounding boxes and class probabilities for each cell — the core idea behind YOLO.

## What I learned

- How YOLO encodes detections as a fixed-size tensor — a 13×13 grid where each cell predicts B boxes, each box encoded as `[objectness, x, y, w, h, class×80]`, giving a `13×13×170` output
- Why plain MSE loss fails for object detection — ~98% of grid cells are empty so the model collapses to predicting nothing, requiring a weighted custom loss that separates objectness, coordinate, and class terms
- How class imbalance in COCO causes shortcut learning — "person" appears in ~60% of images so the model minimised loss by always predicting person in the centre rather than learning spatial features
- That a decreasing loss curve does not mean a model is learning the task — both train and val loss decreased steadily but the predictions were meaningless
- Why Dense layers destroy spatial information — Flatten + Dense loses the 13×13 grid structure, a 1×1 Conv detection head preserves it
- What Batch Normalisation does in practice — stabilises training by normalising activations between layers, allowing higher learning rates and faster convergence
- The difference between overfitting (train loss much lower than val) and shortcut learning (both losses decrease but predictions are wrong)

## Architecture decisions

- **1×1 Conv detection head** instead of Flatten + Dense — preserves the spatial grid structure needed for YOLO predictions
- **Batch Normalisation** after every Conv layer — stabilises training
- **Dropout(0.3)** on the last two Conv blocks — regularisation to prevent overfitting
- **Custom YOLO loss** with 5 components: coordinate loss (λ=5), size loss (√w,√h), objectness loss (λ=1), no-object loss (λ=0.1), class loss (λ=1)
- **lr=0.0001** — lower learning rate for stable training with custom loss

## Why it failed

The model found a shortcut: predicting "person in the centre" satisfies the loss function reasonably well because person is the most common class and objects tend to be centred. The model learned the dataset statistics rather than how to detect objects. The fix would be to filter to fewer classes (removing the imbalance) or use class-weighted loss.

## Results

- Training ran for 24 of 300 epochs before EarlyStopping triggered
- Final train loss: 0.0369 — Final val loss: 0.0606 — Test loss: 0.0535
- No model saved — weights were not useful for detection
