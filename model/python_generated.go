package model

import (
	"strings"
)

// This code is generated automatically using "go generate ./..." from model/model.py. DO NOT EDIT manually.
func GetPythonCode(customDefinitions []string) string {
	return strings.ReplaceAll(`import json
import os
import logging
import sys

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.disable(logging.WARNING)

custom_objects = {}

# tfkg-custom-definitions

config = json.load(sys.stdin)

model = tf.keras.models.model_from_json(config["model_config"], custom_objects=custom_objects)

if config["model_definition_save_dir"] != "":
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    with open(config["model_definition_save_dir"] + "/model-summary.txt", "w") as f:
        f.write("\n".join(summary))

learn_input_signature = [
    tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
    tf.TensorSpec(shape=None, dtype=tf.float32),
]
predict_input_signature = []

zero_inputs = []

model_config = json.loads(config["model_config"])

for model_layer in model_config["config"]["layers"]:
    if model_layer["class_name"] == "InputLayer":
        input_shape = [1]
        for dim in model_layer["config"]["batch_input_shape"][1:]:
            input_shape.append(dim)
        zero_inputs.append(
            tf.zeros(shape=input_shape, dtype=model_layer["config"]["dtype"])
        )
        learn_input_signature.append(tf.TensorSpec(
            shape=model_layer["config"]["batch_input_shape"],
            dtype=model_layer["config"]["dtype"],
        ))
        predict_input_signature.append(tf.TensorSpec(
            shape=model_layer["config"]["batch_input_shape"],
            dtype=model_layer["config"]["dtype"],
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
        self._global_step.assign_add(1)
        with tf.GradientTape() as tape:
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
        return [self._model(list(inputs))]


print("Initialising model")

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

print("Saving model")

tf.saved_model.save(
    gm,
    config["save_dir"],
    signatures={
        "learn": gm.learn,
        "evaluate": gm.evaluate,
        "predict": gm.predict,
    },
)

print("Completed model base")`, "# tfkg-custom-definitions", strings.Join(customDefinitions, "\n"))
}
