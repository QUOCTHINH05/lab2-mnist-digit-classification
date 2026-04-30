from __future__ import annotations

import abc
from typing import Optional

import numpy as np
import tensorflow as tf


class MNISTClassifier(abc.ABC):
    """Base class for MNIST digit classifiers."""

    def __init__(self):
        self.model: Optional[tf.keras.Model] = None

    @abc.abstractmethod
    def build_model(self) -> tf.keras.Model:
        """Build and return a compiled Keras model."""

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 10, batch_size: int = 128,
              validation_split: float = 0.1) -> tf.keras.callbacks.History:
        """Train the model on the given data."""

        if self.model is None: 
            self.model = self.build_model()
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        return history

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on the test data.

        TODO: Implement this method.
        - Raise RuntimeError if self.model is None.
        - Use the model's evaluate() method to get loss and accuracy.
        - Use the model's predict() method and np.argmax to get predicted labels.
        - Return a dict with keys: "loss", "accuracy", "y_pred".
        """
        if self.model is None:
            raise RuntimeError("Model has not been built or trained yet.")
        evaluation = self.model.evaluate(x_test, y_test, verbose=0)
        y_pred = self.model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return {"loss": evaluation[0], "accuracy": evaluation[1], "y_pred": y_pred}

    def save(self, path: str) -> None:
        """Save the model to the given file path.

        TODO: Implement this method.
        - Raise RuntimeError if self.model is None.
        - Use the model's save() method.
        """
        if self.model is None:
            raise RuntimeError("Model has not been built or trained yet.")
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load a model from the given file path.

        TODO: Implement this method.
        - Use tf.keras.models.load_model() and assign to self.model.
        """
        self.model = tf.keras.models.load_model(path)


class LogisticRegressionClassifier(MNISTClassifier):
    """Logistic regression (single dense layer with softmax)."""

    def build_model(self) -> tf.keras.Model:
        """Build a logistic regression model for MNIST.

        TODO: Implement this method.
        - Create a Sequential model with:
          - Input layer accepting 784-dimensional vectors.
          - A single Dense output layer with 10 units and softmax activation.
        - Compile with optimizer="sgd", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model


class NeuralNetworkClassifier(MNISTClassifier):
    """Simple feedforward neural network."""

    def build_model(self) -> tf.keras.Model:
        """Build a simple neural network for MNIST.

        TODO: Implement this method.
        - Create a Sequential model with:
          - Input layer accepting 784-dimensional vectors.
          - Dense hidden layer with 128 units and ReLU activation.
          - Dense hidden layer with 64 units and ReLU activation.
          - Dense output layer with 10 units and softmax activation.
        - Compile with optimizer="adam", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model