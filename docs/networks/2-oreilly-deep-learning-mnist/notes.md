# Deep Learning With Keras (O'Reilly) - MNIST CNN

This network started as the TensorFlow Keras counterpart to the NumPy MNIST notebook (Network 1). Same dataset, same task, but built with TensorFlow/Keras instead of writing forward and backward propagation by hand — and this time using a small convolutional network rather than a plain MLP.

## What I learned

- How convolutional layers learn image features instead of treating pixels as independent features, and why they beat a dense-only network on the same task
- How `MaxPooling2D` shrinks the spatial dimensions between conv blocks to keep parameter counts manageable
- How `validation_split` gives you an honest generalisation signal *during* training, without touching the test set
- How `EarlyStopping` uses that validation signal to halt training automatically once `val_loss` plateaus, preventing overfitting without having to hand-tune the epoch count

## Things to try next

- Add `BatchNormalization` layers after each `Conv2D` to see how much faster it converges
- Try simple data augmentation (random rotation, translation) and see if it closes the remaining ~0.5% train/test gap
- Retrain on Fashion-MNIST using the exact same architecture as a generalisation test
