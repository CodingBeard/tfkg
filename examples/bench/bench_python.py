import os
import time

import tensorflow as tf
import numpy as np

x = []
y = []
for i in range(0, 29000):
    x.append(range(0, 1000))
    if i % 2 == 0:
        y.append(0)
    else:
        y.append(1)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1000, ), dtype=tf.float32),
    tf.keras.layers.Embedding(1000, 32, input_length=1000),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(1024, activation="swish"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[
        "accuracy",
        tf.keras.metrics.SpecificityAtSensitivity(0.99, name="specAtSen99"),
    ]
)

start = int(time.time())

model.fit(
    np.array(x, dtype="float32"),
    np.array(y),
    batch_size=500,
    epochs=1,
    verbose=1,
)

end = int(time.time())

print(end-start, "s - python")
