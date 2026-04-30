# Lab Assignment: MNIST Digit Classification with TensorFlow

## 1. Objective

In this lab you will implement two machine-learning classifiers for handwritten digit recognition on the **MNIST** dataset using **TensorFlow / Keras**:

1. **Logistic Regression** — a single-layer linear model.
2. **Simple Neural Network** — a feedforward network with two hidden layers.

You will work with a provided codebase that already contains the full training/testing pipeline (`main.py`). Your task is to **complete the implementation** of the classifier classes in `classifier.py`, then execute the pipeline and report your results.

---

## 2. Provided Materials

You will receive a project folder with the following structure:

```
mnist/
├── main.py                  # Complete – DO NOT MODIFY
├── classifier.py            # Stub – YOU IMPLEMENT THIS
├── data/
│   └── mnist.npz            # Pre-split dataset (60 000 train / 10 000 test)
├── models/                  # Reference trained models (teacher's results)
│   ├── logistic_model.keras
│   └── nn_model.keras
└── results/                 # Reference results (teacher's results)
    ├── logistic_results.md
    ├── logistic_results.json
    ├── nn_results.md
    ├── nn_results.json
    └── summary.md
```

| File / Folder | Description |
|---|---|
| `main.py` | CLI program with commands: `configure`, `train`, `test`, `summary`. **Do not modify.** |
| `classifier.py` | Contains the base class `MNISTClassifier` and two child classes with method stubs. **This is the file you will implement.** |
| `data/mnist.npz` | Pre-processed MNIST dataset. Training and test sets are fixed for consistent comparison. |
| `models/` | Reference models produced by the teacher. Use for comparison only. |
| `results/` | Reference evaluation results. Your results should be close to these values. |

---

## 3. Prerequisites

Install the required Python packages before starting:

```bash
pip install tensorflow numpy scikit-learn
```

> **Note:** TensorFlow 2.15.x is recommended for Python 3.9. For Python 3.10+, TensorFlow 2.16+ should work.

---

## 4. Tasks

### Task 1 — Implement `MNISTClassifier` Base Class Methods

Open `classifier.py`. The base class `MNISTClassifier` has four methods marked with `TODO`. Implement each one:

| Method | Description |
|---|---|
| `train(...)` | If the model has not been built yet, call `build_model()`. Then call `model.fit()` with the provided parameters and return the `History` object. |
| `evaluate(...)` | Call `model.evaluate()` to obtain loss and accuracy. Call `model.predict()` and use `np.argmax` to get predicted labels. Return a dictionary with keys `"loss"`, `"accuracy"`, `"y_pred"`. |
| `save(path)` | Save the Keras model to the given path using `model.save()`. |
| `load(path)` | Load a Keras model from the given path using `tf.keras.models.load_model()`. |

### Task 2 — Implement `LogisticRegressionClassifier.build_model()`

Build and compile a Keras `Sequential` model that represents logistic regression:

- **Input:** 784-dimensional vector (flattened 28×28 image).
- **Output layer:** Dense layer with **10 units** and **softmax** activation.
- **Optimizer:** `sgd`
- **Loss:** `sparse_categorical_crossentropy`
- **Metrics:** `accuracy`

### Task 3 — Implement `NeuralNetworkClassifier.build_model()`

Build and compile a Keras `Sequential` model with hidden layers:

- **Input:** 784-dimensional vector.
- **Hidden layer 1:** Dense, **128 units**, **ReLU** activation.
- **Hidden layer 2:** Dense, **64 units**, **ReLU** activation.
- **Output layer:** Dense, **10 units**, **softmax** activation.
- **Optimizer:** `adam`
- **Loss:** `sparse_categorical_crossentropy`
- **Metrics:** `accuracy`

### Task 4 — Execute the Pipeline

Run the following commands **in order**:

```bash
# Step 1: Train the logistic regression model
python main.py train --model logistic

# Step 2: Test the logistic regression model
python main.py test --model logistic

# Step 3: Train the neural network model
python main.py train --model nn

# Step 4: Test the neural network model
python main.py test --model nn

# Step 5: Generate the summary report
python main.py summary
```

> **Note:** The `data/mnist.npz` file is already provided. You do **not** need to run `python main.py configure`.

### Task 5 — Write a Report

Prepare a short report (1–2 pages) that includes:

1. Your `results/summary.md` table (copy it into your report).
2. A comparison of the two models:
   - Which model achieved higher accuracy? By how much?
   - How do the number of weights (parameters) compare between the two models?
   - What is the trade-off between model complexity and accuracy?
3. Identify which digit(s) each model struggles with the most (refer to the per-class classification report or confusion matrix).
4. **Bonus:** Suggest one improvement to the neural network architecture and briefly explain why it might help.

---

## 5. Reference Results

Your results should be close to the values below. Small variations are acceptable due to random weight initialization.

| Model | Weights | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| logistic | 7,850 | 0.9127 | 0.9126 | 0.9127 | 0.9124 |
| nn | 109,386 | 0.9773 | 0.9775 | 0.9773 | 0.9773 |

---

## 6. Submission

Submit the following files in [Student ID].zip:

1. **`classifier.py`** — Your completed implementation.
2. **`results/summary.md`** — Generated summary table.
3. **`results/logistic_results.json`** — Logistic regression JSON results.
4. **`results/nn_results.json`** — Neural network JSON results.
5. **Report** (PDF or Markdown) — Your written analysis.

---

## 7. Grading Rubric

| Criteria | Points |
|---|---|
| `MNISTClassifier` base class methods implemented correctly | 20 |
| `LogisticRegressionClassifier.build_model()` implemented correctly | 15 |
| `NeuralNetworkClassifier.build_model()` implemented correctly | 15 |
| Pipeline executes without errors | 15 |
| Results are within acceptable range of reference values | 15 |
| Report: comparison and analysis | 15 |
| Report: bonus improvement suggestion | 5 |
| **Total** | **100** |

---

## 8. Tips

- Read the docstrings in `classifier.py` carefully — they describe exactly what each method should do.
- Study `main.py` to understand how your classifier methods will be called.
- Use the reference results in the `results/` folder to verify your implementation is correct.
- If your accuracy is significantly lower than the reference, double-check your model architecture (layer sizes, activations) and compilation settings (optimizer, loss).
- You can inspect the reference models with:
  ```python
  import tensorflow as tf
  model = tf.keras.models.load_model("models/logistic_model.keras")
  model.summary()
  ```
