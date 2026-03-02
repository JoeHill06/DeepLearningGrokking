# Deep Learning from Scratch

Neural network implementations built from scratch in NumPy, following along with **[Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning)** by Andrew W. Trask (Manning Publications).

The goal is to understand the mechanics of deep learning by implementing everything by hand — no high-level frameworks for the actual learning logic.

---

## Examples

### `trafficLights.py` — Street Light Classifier

A minimal two-layer network that learns to predict walk/stop from streetlight patterns.

**Features used:**
- Forward propagation through two weight layers
- Mean squared error loss
- Backpropagation with hand-computed deltas
- ReLU activation (`relu`, `relu2deriv`)
- Stochastic gradient descent (one sample at a time)

**Architecture:** `3 inputs → 4 hidden → 1 output`

---

### `mnist.py` — Handwritten Digit Classifier

A two-layer network trained on the MNIST dataset (1000 training images) that classifies digits 0–9.

**Features used:**
- Mini-batch gradient descent (`batch_size = 100`)
- Dropout regularization (50% drop rate, rescaled by 2 to preserve expected values)
- ReLU activation
- One-hot encoded labels
- Train/test split evaluation printed every 10 iterations

**Architecture:** `784 inputs → 100 hidden → 10 output`

**Performance:** ~85% train accuracy, ~80% test accuracy at iteration 290

---

## Concepts Covered (from Grokking Deep Learning)

| Concept | Where used |
|---|---|
| Gradient descent | Both examples |
| Backpropagation | Both examples |
| ReLU & its derivative | Both examples |
| Mini-batch training | `mnist.py` |
| Dropout regularization | `mnist.py` |
| One-hot encoding | `mnist.py` |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
python trafficLights.py
python mnist.py
```
