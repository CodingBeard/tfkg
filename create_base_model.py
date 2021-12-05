import tensorflow as tf
import tensorflow.keras as k


class GolangModel(tf.Module):
    def __init__(self):
        super().__init__()

        bool_input = k.layers.Input(
            shape=(3,),
            name='input',
            dtype=tf.float32,
            batch_size=10
        )

        output = k.layers.Dense(
            1,
            name="output",
            dtype=tf.float32,
        )(bool_input)

        self.model = k.Model(bool_input, output)
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self._optimizer = k.optimizers.Adam()
        self._loss = k.losses.binary_crossentropy

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        ]
    )
    def learn(self, data, labels):
        self._global_step.assign_add(1)
        with tf.GradientTape() as tape:
            loss = self._loss(labels, self.model(data))

        gradient = tape.gradient(loss, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        return {"loss": loss}

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)])
    def predict(self, data):
        prediction = self.model(data)
        return {"prediction": prediction}


gm = GolangModel()

gm.learn(
    tf.zeros([10, 3], dtype=tf.float32),
    tf.zeros([10, 1], dtype=tf.float32),
)
gm.predict(tf.zeros((10, 3), dtype=tf.float32))

tf.saved_model.save(
    gm,
    "base_model",
    signatures={
        "learn": gm.learn,
        "predict": gm.predict,
    },
)

with open("base_model/graph_def_learn.graph", "wb") as f:
    f.write(tf.function(gm.learn).get_concrete_function(
        tf.zeros([10, 3], dtype=tf.float32),
        tf.zeros([10, 1], dtype=tf.float32),
    ).graph.as_graph_def().SerializeToString())

with open("base_model/graph_def_predict.graph", "wb") as f:
    f.write(tf.function(gm.predict).get_concrete_function(
        tf.zeros((10, 3), dtype=tf.float32)
    ).graph.as_graph_def().SerializeToString())

