import os
import tensorflow as tf
import numpy as np

csv_data = np.loadtxt('data/iris.data', delimiter=',')
target_all = csv_data[:, -1]

csv_data = csv_data[:, 0:-1]

shuffled_indices = np.arange(csv_data.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = csv_data[shuffled_indices]
shuffled_targets = target_all[shuffled_indices]

train_inputs = shuffled_inputs
train_targets = shuffled_targets

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,), dtype=tf.float32),
    tf.keras.layers.Dense(10, activation="swish"),
    tf.keras.layers.Dense(10, activation="swish"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)

model.fit(
    train_inputs,
    train_targets,
    batch_size=3,
    epochs=10,
    verbose=0,
)

model.evaluate(
    train_inputs,
    train_targets,
    batch_size=3,
    verbose=1,
)

model.save("vanilla_model")
