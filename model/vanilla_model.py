import json
import os
import logging
import sys

import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.disable(logging.WARNING)

with open(sys.argv[1], "r") as f:
    config = json.load(f)

print("Loading Vanilla model")

model = tf.keras.models.load_model(config["model_dir"])

learn_input_signature = [
    tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
    tf.TensorSpec(shape=None, dtype=tf.float32),
]
predict_input_signature = []

zero_inputs = []

for model_layer in model.inputs:
    input_shape = [1]
    for dim in model_layer.shape[1:]:
        input_shape.append(dim)
    zero_inputs.append(
        tf.zeros(shape=input_shape, dtype=model_layer.dtype)
    )
    learn_input_signature.append(tf.TensorSpec(
        shape=model_layer.shape,
        dtype=model_layer.dtype,
    ))
    predict_input_signature.append(tf.TensorSpec(
        shape=model_layer.shape,
        dtype=model_layer.dtype,
    ))

evaluate_input_signature = learn_input_signature


class GolangModel(tf.Module):
    def __init__(self):
        super().__init__()

        self._model = model

        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        opt = tf.keras.optimizers.get(config["optimizer"]["class_name"])
        self._optimizer = opt.from_config(config["optimizer"]["config"])
        loss_func = None
        if config["loss"] == "binary_crossentropy":
            loss_func = tf.keras.losses.BinaryCrossentropy(reduction="none")
        elif config["loss"] == "sparse_categorical_crossentropy":
            loss_func = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")

        def loss(y_true, y_pred, class_weights):
            weighted_loss = tf.multiply(
                loss_func(y_true, y_pred),
                class_weights
            )
            return tf.reduce_mean(weighted_loss)

        self._loss = loss

    @tf.function(input_signature=learn_input_signature)
    def learn(
            self,
            y,
            class_weights,
            *inputs
    ):
        self._global_step.assign_add(1)
        with tf.GradientTape() as tape:
            logits = self._model(list(inputs), training=True)
            loss = self._loss(y, logits, class_weights)

        gradient = tape.gradient(
            loss,
            self._model.trainable_variables
        )
        self._optimizer.apply_gradients(
            zip(gradient, self._model.trainable_variables)
        )
        return [
            loss,
            logits
        ]

    @tf.function(input_signature=evaluate_input_signature)
    def evaluate(
            self,
            y,
            class_weights,
            *inputs
    ):
        logits = self._model(list(inputs), training=False)
        loss = self._loss(y, logits, class_weights)

        return [
            loss,
            logits
        ]

    @tf.function(input_signature=predict_input_signature)
    def predict(
            self,
            *inputs,
    ):
        return [self._model(list(inputs), training=False)]

    @tf.function(input_signature=[])
    def get_weights(
            self,
    ):
        return self._model.weights


gm = GolangModel()

y_zeros = tf.zeros(shape=[1, 1], dtype=tf.int32)
class_weights_ones = tf.ones(shape=1, dtype=tf.float32)

print("Tracing learn")

gm.learn(
    y_zeros,
    class_weights_ones,
    *zero_inputs,
)

print("Tracing evaluate")

gm.evaluate(
    y_zeros,
    class_weights_ones,
    *zero_inputs,
)

print("Tracing predict")

gm.predict(*zero_inputs)

print("Tracing get_weights")

gm.get_weights()

print("Saving model")

tf.saved_model.save(
    gm,
    config["save_dir"],
    signatures={
        "learn": gm.learn,
        "evaluate": gm.evaluate,
        "predict": gm.predict,
        "get_weights": gm.get_weights,
    },
)

print("Completed model base")