import inspect
import json

import tensorflow.keras as k
import numpy as np

objects = [
    {"type": "initializer", "class": k.initializers.RandomNormal, "args": []},
    {"type": "initializer", "class": k.initializers.RandomUniform, "args": []},
    {"type": "initializer", "class": k.initializers.TruncatedNormal, "args": []},
    {"type": "initializer", "class": k.initializers.Zeros, "args": []},
    {"type": "initializer", "class": k.initializers.Ones, "args": []},
    {"type": "initializer", "class": k.initializers.GlorotNormal, "args": []},
    {"type": "initializer", "class": k.initializers.GlorotUniform, "args": []},
    {"type": "initializer", "class": k.initializers.HeNormal, "args": []},
    {"type": "initializer", "class": k.initializers.HeUniform, "args": []},
    {"type": "initializer", "class": k.initializers.Identity, "args": []},
    {"type": "initializer", "class": k.initializers.Orthogonal, "args": []},
    {"type": "initializer", "class": k.initializers.Constant, "args": []},
    {"type": "initializer", "class": k.initializers.VarianceScaling, "args": []},
    {"type": "regularizer", "class": k.regularizers.l1, "args": []},
    {"type": "regularizer", "class": k.regularizers.l2, "args": []},
    {"type": "constraint", "class": k.constraints.MaxNorm, "args": []},
    {"type": "constraint", "class": k.constraints.MinMaxNorm, "args": []},
    {"type": "constraint", "class": k.constraints.NonNeg, "args": []},
    {"type": "constraint", "class": k.constraints.UnitNorm, "args": []},
    {"type": "constraint", "class": k.constraints.RadialConstraint, "args": []},
    {"type": "layer", "class": k.layers.Dense, "args": []},
    {"type": "layer", "class": k.layers.Activation, "args": []},
    {"type": "layer", "class": k.layers.Embedding, "args": []},
    {"type": "layer", "class": k.layers.Masking, "args": []},
    {"type": "layer", "class": k.layers.Conv1D, "args": []},
    {"type": "layer", "class": k.layers.Conv2D, "args": []},
    {"type": "layer", "class": k.layers.Conv3D, "args": []},
    {"type": "layer", "class": k.layers.SeparableConv1D, "args": []},
    {"type": "layer", "class": k.layers.SeparableConv2D, "args": []},
    {"type": "layer", "class": k.layers.DepthwiseConv2D, "args": []},
    {"type": "layer", "class": k.layers.Conv2DTranspose, "args": []},
    {"type": "layer", "class": k.layers.Conv3DTranspose, "args": []},
    {"type": "layer", "class": k.layers.MaxPooling1D, "args": []},
    {"type": "layer", "class": k.layers.MaxPooling2D, "args": []},
    {"type": "layer", "class": k.layers.MaxPooling3D, "args": []},
    {"type": "layer", "class": k.layers.AveragePooling1D, "args": []},
    {"type": "layer", "class": k.layers.AveragePooling2D, "args": []},
    {"type": "layer", "class": k.layers.AveragePooling3D, "args": []},
    {"type": "layer", "class": k.layers.GlobalMaxPooling1D, "args": []},
    {"type": "layer", "class": k.layers.GlobalMaxPooling2D, "args": []},
    {"type": "layer", "class": k.layers.GlobalMaxPooling3D, "args": []},
    {"type": "layer", "class": k.layers.GlobalAveragePooling1D, "args": []},
    {"type": "layer", "class": k.layers.GlobalAveragePooling2D, "args": []},
    {"type": "layer", "class": k.layers.GlobalAveragePooling3D, "args": []},
    {"type": "layer", "class": k.layers.LSTM, "args": []},
    {"type": "layer", "class": k.layers.GRU, "args": []},
    {"type": "layer", "class": k.layers.SimpleRNN, "args": []},
    {"type": "layer", "class": k.layers.TimeDistributed, "args": []},
    {"type": "layer", "class": k.layers.Bidirectional, "args": []},
    {"type": "layer", "class": k.layers.ConvLSTM2D, "args": []},
    {"type": "layer", "class": k.layers.BatchNormalization, "args": []},
    {"type": "layer", "class": k.layers.LayerNormalization, "args": []},
    {"type": "layer", "class": k.layers.Dropout, "args": []},
    {"type": "layer", "class": k.layers.SpatialDropout1D, "args": []},
    {"type": "layer", "class": k.layers.SpatialDropout2D, "args": []},
    {"type": "layer", "class": k.layers.SpatialDropout3D, "args": []},
    {"type": "layer", "class": k.layers.GaussianDropout, "args": []},
    {"type": "layer", "class": k.layers.GaussianNoise, "args": []},
    {"type": "layer", "class": k.layers.ActivityRegularization, "args": []},
    {"type": "layer", "class": k.layers.AlphaDropout, "args": []},
    {"type": "layer", "class": k.layers.MultiHeadAttention, "args": []},
    {"type": "layer", "class": k.layers.Attention, "args": []},
    {"type": "layer", "class": k.layers.AdditiveAttention, "args": []},
    {"type": "layer", "class": k.layers.Reshape, "args": []},
    {"type": "layer", "class": k.layers.Flatten, "args": []},
    {"type": "layer", "class": k.layers.RepeatVector, "args": []},
    {"type": "layer", "class": k.layers.Permute, "args": []},
    {"type": "layer", "class": k.layers.Cropping1D, "args": []},
    {"type": "layer", "class": k.layers.Cropping2D, "args": []},
    {"type": "layer", "class": k.layers.Cropping3D, "args": []},
    {"type": "layer", "class": k.layers.UpSampling1D, "args": []},
    {"type": "layer", "class": k.layers.UpSampling2D, "args": []},
    {"type": "layer", "class": k.layers.UpSampling3D, "args": []},
    {"type": "layer", "class": k.layers.ZeroPadding1D, "args": []},
    {"type": "layer", "class": k.layers.ZeroPadding2D, "args": []},
    {"type": "layer", "class": k.layers.ZeroPadding3D, "args": []},
    {"type": "layer", "class": k.layers.Concatenate, "args": []},
    {"type": "layer", "class": k.layers.Average, "args": []},
    {"type": "layer", "class": k.layers.Maximum, "args": []},
    {"type": "layer", "class": k.layers.Minimum, "args": []},
    {"type": "layer", "class": k.layers.Add, "args": []},
    {"type": "layer", "class": k.layers.Subtract, "args": []},
    {"type": "layer", "class": k.layers.Multiply, "args": []},
    {"type": "layer", "class": k.layers.Dot, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.CategoryCrossing, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.CategoryEncoding, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.CenterCrop, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.Discretization, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.Hashing, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.IntegerLookup, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.Normalization, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.PreprocessingLayer, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.RandomContrast, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.RandomCrop, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.RandomFlip, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.RandomHeight, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.RandomRotation, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.RandomTranslation, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.RandomWidth, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.RandomZoom, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.Rescaling, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.Resizing, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.StringLookup, "args": []},
    {"type": "layer", "class": k.layers.experimental.preprocessing.TextVectorization, "args": []},
    {"type": "layer", "class": k.layers.experimental.EinsumDense, "args": []},
    {"type": "layer", "class": k.layers.experimental.RandomFourierFeatures, "args": []},
    {"type": "layer", "class": k.layers.experimental.SyncBatchNormalization, "args": []},
    {"type": "optimizer", "class": k.optimizers.Adagrad, "args": []},
    {"type": "optimizer", "class": k.optimizers.Adadelta, "args": []},
    {"type": "optimizer", "class": k.optimizers.Adam, "args": []},
    {"type": "optimizer", "class": k.optimizers.Adamax, "args": []},
    {"type": "optimizer", "class": k.optimizers.Ftrl, "args": []},
    {"type": "optimizer", "class": k.optimizers.Nadam, "args": []},
    {"type": "optimizer", "class": k.optimizers.RMSprop, "args": []},
    {"type": "optimizer", "class": k.optimizers.SGD, "args": []},
]
defaults = {
    "activation": "linear",
    "layer": k.layers.LSTM(1),
    "target_shape": (1,),
    "dims": (1,),
}
configs = []
for obj in objects:
    c = obj["class"]
    a = obj["args"]
    sig = inspect.signature(c.__init__)
    required_params = []
    optional_params = []
    for param_name in sig.parameters:
        if param_name == "self" or param_name == "args" or param_name == "kwargs":
            continue
        param = sig.parameters[param_name]
        if param.default.__class__.__name__ == "type" and param.kind.name != "VAR_KEYWORD":
            required_param = [param_name]
            if param_name in defaults:
                if param_name != "layer":
                    required_param.append(defaults[param_name])
                else:
                    required_param.append(None)
                a.append(defaults[param_name])
            else:
                required_param.append(1)
                a.append(1)
            required_params.append(required_param)
        else:
            if param_name == "num_tokens":
                a.append(1)
            optional_params.append([param_name, param.default])
    print(c.__name__)
    sub_config = c(*a).get_config()
    object_config = {
        "class_name": c.__name__,
        "config": sub_config,
    }
    if obj["type"] == "layer":
        object_config["input_nodes"] = [["", 0, 0, None]]

    if "name" in sub_config:
        object_config["name"] = sub_config["name"]
    configs.append({
        "type": obj["type"],
        "name": c.__name__,
        "required": required_params,
        "optional": optional_params,
        "config": object_config,
    })


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


with open("generate/python/objects.json", "w") as f:
    json.dump(configs, f, indent=2, cls=NpEncoder)

