import json
import os
import logging
import sys

import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.disable(logging.WARNING)

custom_objects = {}

# tfkg-custom-definitions

with open(sys.argv[1], "r") as f:
    config = json.load(f)

model = tf.keras.models.model_from_json(config["model_config"], custom_objects=custom_objects)

weights_spec = []
for item in model.weights:
    weights_spec.append(tf.TensorSpec(shape=item.shape, dtype=item.dtype))

if config["model_definition_save_dir"] != "":
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    with open(config["model_definition_save_dir"] + "/model-summary.txt", "w") as f:
        f.write("\n".join(summary))
    weight_names = []
    for item in model.weights:
        weight_names.append(item.name)
    with open(config["model_definition_save_dir"] + "/weight_names.json", "w") as f:
        json.dump(weight_names, f)

learn_signature = []
predict_input_signature = []

zero_inputs = []

model_config = json.loads(config["model_config"])

for model_layer in model_config["config"]["layers"]:
    if model_layer["class_name"] == "InputLayer":
        input_shape = [config["batch_size"]]
        predict_input_shape = [None]
        for dim in model_layer["config"]["batch_input_shape"][1:]:
            input_shape.append(dim)
            predict_input_shape.append(dim)
        zero_inputs.append(
            tf.zeros(shape=input_shape, dtype=model_layer["config"]["dtype"])
        )
        learn_signature.append(tf.TensorSpec(
            shape=input_shape,
            dtype=model_layer["config"]["dtype"],
        ))
        predict_input_signature.append(tf.TensorSpec(
            shape=predict_input_shape,
            dtype=model_layer["config"]["dtype"],
        ))

last_layer = model_config["config"]["layers"][-1]

y_dtype = model.output.dtype
y_shape = model.output.shape

if config["loss"] == "binary_crossentropy" or config["loss"] == "sparse_categorical_crossentropy":
    y_dtype = tf.int32
    y_shape = [config["batch_size"], 1]

learn_input_signature = [
    tf.TensorSpec(shape=y_shape, dtype=y_dtype),
    tf.TensorSpec(shape=config["batch_size"], dtype=tf.float32),
]
for sig in learn_signature:
    learn_input_signature.append(sig)

evaluate_input_signature = learn_input_signature


class GolangModel(tf.Module):
    def __init__(self):
        super().__init__()

        self._model = model

        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        opt = tf.keras.optimizers.get(config["optimizer"]["class_name"])
        self._optimizer = opt.from_config(config["optimizer"]["config"])
        if config["loss"] == "binary_crossentropy":
            loss_func = tf.keras.losses.BinaryCrossentropy(reduction="none")

            def loss(y_true, y_pred, class_weights):
                weighted_loss = tf.multiply(
                    loss_func(y_true, y_pred),
                    class_weights
                )
                return tf.reduce_mean(weighted_loss)

            self._loss = loss
        elif config["loss"] == "sparse_categorical_crossentropy":
            loss_func = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")

            def loss(y_true, y_pred, class_weights):
                weighted_loss = tf.multiply(
                    loss_func(y_true, y_pred),
                    class_weights
                )
                return tf.reduce_mean(weighted_loss)

            self._loss = loss
        elif config["loss"] == "mse":
            loss_func = tf.keras.losses.MeanSquaredError(reduction="none")

            def loss(y_true, y_pred, class_weights):
                return tf.reduce_mean(loss_func(y_true, y_pred))

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
            logits = self._model(inputs, training=True)
            loss = self._loss(y, logits, class_weights)

        self._optimizer.minimize(loss, self._model.trainable_variables, tape=tape)
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

    @tf.function(input_signature=weights_spec)
    def set_weights(
            self,
            *weights,
    ):
        for i in range(0, len(self._model.weights)):
            self._model.weights[i].assign(weights[i])

        return weights


print("Initialising model")

gm = GolangModel()

output_shape = [config["batch_size"]]
for dim in y_shape[1:]:
    output_shape.append(dim)

y_zeros = tf.zeros(shape=output_shape, dtype=y_dtype)
class_weights_ones = tf.ones(shape=config["batch_size"], dtype=tf.float32)

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

print("Tracing set_weights")

ws = gm.get_weights()
gm.set_weights(*ws)

print("Saving model")

tf.saved_model.save(
    gm,
    config["save_dir"],
    signatures={
        "learn": gm.learn,
        "evaluate": gm.evaluate,
        "predict": gm.predict,
        "set_weights": gm.set_weights,
    },
)

print("Completed model base")