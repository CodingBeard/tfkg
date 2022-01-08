package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LcuDNNLSTM struct {
	activation           string
	activityRegularizer  regularizer.Regularizer
	biasConstraint       constraint.Constraint
	biasInitializer      initializer.Initializer
	biasRegularizer      regularizer.Regularizer
	dropout              float64
	dtype                DataType
	goBackwards          bool
	implementation       float64
	inputs               []Layer
	kernelConstraint     constraint.Constraint
	kernelInitializer    initializer.Initializer
	kernelRegularizer    regularizer.Regularizer
	name                 string
	recurrentActivation  string
	recurrentConstraint  constraint.Constraint
	recurrentDropout     float64
	recurrentInitializer initializer.Initializer
	recurrentRegularizer regularizer.Regularizer
	returnSequences      bool
	returnState          bool
	shape                tf.Shape
	stateful             bool
	timeMajor            bool
	trainable            bool
	unitForgetBias       bool
	units                float64
	unroll               bool
	useBias              bool
	layerWeights         []*tf.Tensor
}

// CuDNNLSTM if trained on a GPU, a GPU is required for inference unless you compile the model with CpuInference: true option
func CuDNNLSTM(units float64) *LcuDNNLSTM {
	return &LcuDNNLSTM{
		activation:           "tanh",
		activityRegularizer:  &regularizer.NilRegularizer{},
		biasConstraint:       &constraint.NilConstraint{},
		biasInitializer:      initializer.Zeros(),
		biasRegularizer:      &regularizer.NilRegularizer{},
		dropout:              0,
		dtype:                Float32,
		goBackwards:          false,
		implementation:       2,
		kernelConstraint:     &constraint.NilConstraint{},
		kernelInitializer:    initializer.GlorotUniform(),
		kernelRegularizer:    &regularizer.NilRegularizer{},
		name:                 UniqueName("lstm_1"),
		recurrentActivation:  "sigmoid",
		recurrentConstraint:  &constraint.NilConstraint{},
		recurrentDropout:     0,
		recurrentInitializer: initializer.Orthogonal(),
		recurrentRegularizer: &regularizer.NilRegularizer{},
		returnSequences:      false,
		returnState:          false,
		stateful:             false,
		timeMajor:            false,
		trainable:            true,
		unitForgetBias:       true,
		units:                units,
		unroll:               false,
		useBias:              true,
	}
}

func (l *LcuDNNLSTM) SetActivation(activation string) *LcuDNNLSTM {
	l.activation = activation
	return l
}

func (l *LcuDNNLSTM) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LcuDNNLSTM {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LcuDNNLSTM) SetBiasConstraint(biasConstraint constraint.Constraint) *LcuDNNLSTM {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LcuDNNLSTM) SetBiasInitializer(biasInitializer initializer.Initializer) *LcuDNNLSTM {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LcuDNNLSTM) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LcuDNNLSTM {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LcuDNNLSTM) SetDropout(dropout float64) *LcuDNNLSTM {
	l.dropout = dropout
	return l
}

func (l *LcuDNNLSTM) SetDtype(dtype DataType) *LcuDNNLSTM {
	l.dtype = dtype
	return l
}

func (l *LcuDNNLSTM) SetGoBackwards(goBackwards bool) *LcuDNNLSTM {
	l.goBackwards = goBackwards
	return l
}

func (l *LcuDNNLSTM) SetImplementation(implementation float64) *LcuDNNLSTM {
	l.implementation = implementation
	return l
}

func (l *LcuDNNLSTM) SetKernelConstraint(kernelConstraint constraint.Constraint) *LcuDNNLSTM {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LcuDNNLSTM) SetKernelInitializer(kernelInitializer initializer.Initializer) *LcuDNNLSTM {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LcuDNNLSTM) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LcuDNNLSTM {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LcuDNNLSTM) SetName(name string) *LcuDNNLSTM {
	l.name = name
	return l
}

func (l *LcuDNNLSTM) SetRecurrentActivation(recurrentActivation string) *LcuDNNLSTM {
	l.recurrentActivation = recurrentActivation
	return l
}

func (l *LcuDNNLSTM) SetRecurrentConstraint(recurrentConstraint constraint.Constraint) *LcuDNNLSTM {
	l.recurrentConstraint = recurrentConstraint
	return l
}

func (l *LcuDNNLSTM) SetRecurrentDropout(recurrentDropout float64) *LcuDNNLSTM {
	l.recurrentDropout = recurrentDropout
	return l
}

func (l *LcuDNNLSTM) SetRecurrentInitializer(recurrentInitializer initializer.Initializer) *LcuDNNLSTM {
	l.recurrentInitializer = recurrentInitializer
	return l
}

func (l *LcuDNNLSTM) SetRecurrentRegularizer(recurrentRegularizer regularizer.Regularizer) *LcuDNNLSTM {
	l.recurrentRegularizer = recurrentRegularizer
	return l
}

func (l *LcuDNNLSTM) SetReturnSequences(returnSequences bool) *LcuDNNLSTM {
	l.returnSequences = returnSequences
	return l
}

func (l *LcuDNNLSTM) SetReturnState(returnState bool) *LcuDNNLSTM {
	l.returnState = returnState
	return l
}

func (l *LcuDNNLSTM) SetShape(shape tf.Shape) *LcuDNNLSTM {
	l.shape = shape
	return l
}

func (l *LcuDNNLSTM) SetStateful(stateful bool) *LcuDNNLSTM {
	l.stateful = stateful
	return l
}

func (l *LcuDNNLSTM) SetTimeMajor(timeMajor bool) *LcuDNNLSTM {
	l.timeMajor = timeMajor
	return l
}

func (l *LcuDNNLSTM) SetTrainable(trainable bool) *LcuDNNLSTM {
	l.trainable = trainable
	return l
}

func (l *LcuDNNLSTM) SetUnitForgetBias(unitForgetBias bool) *LcuDNNLSTM {
	l.unitForgetBias = unitForgetBias
	return l
}

func (l *LcuDNNLSTM) SetUnroll(unroll bool) *LcuDNNLSTM {
	l.unroll = unroll
	return l
}

func (l *LcuDNNLSTM) SetUseBias(useBias bool) *LcuDNNLSTM {
	l.useBias = useBias
	return l
}

func (l *LcuDNNLSTM) SetLayerWeights(layerWeights []*tf.Tensor) *LcuDNNLSTM {
	l.layerWeights = layerWeights
	return l
}

func (l *LcuDNNLSTM) GetShape() tf.Shape {
	return l.shape
}

func (l *LcuDNNLSTM) GetDtype() DataType {
	return l.dtype
}

func (l *LcuDNNLSTM) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LcuDNNLSTM) GetInputs() []Layer {
	return l.inputs
}

func (l *LcuDNNLSTM) GetName() string {
	return l.name
}

func (l *LcuDNNLSTM) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLGpuLSTM struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LcuDNNLSTM) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range l.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigLGpuLSTM{
		ClassName: "GpuLSTM",
		Name:      l.name,
		Config: map[string]interface{}{
			"activation":            l.activation,
			"activity_regularizer":  l.activityRegularizer.GetKerasLayerConfig(),
			"bias_constraint":       l.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":      l.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":      l.biasRegularizer.GetKerasLayerConfig(),
			"dropout":               l.dropout,
			"dtype":                 l.dtype.String(),
			"go_backwards":          l.goBackwards,
			"implementation":        l.implementation,
			"kernel_constraint":     l.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":    l.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":    l.kernelRegularizer.GetKerasLayerConfig(),
			"name":                  l.name,
			"recurrent_activation":  l.recurrentActivation,
			"recurrent_constraint":  l.recurrentConstraint.GetKerasLayerConfig(),
			"recurrent_dropout":     l.recurrentDropout,
			"recurrent_initializer": l.recurrentInitializer.GetKerasLayerConfig(),
			"recurrent_regularizer": l.recurrentRegularizer.GetKerasLayerConfig(),
			"return_sequences":      l.returnSequences,
			"return_state":          l.returnState,
			"stateful":              l.stateful,
			"time_major":            l.timeMajor,
			"trainable":             l.trainable,
			"unit_forget_bias":      l.unitForgetBias,
			"units":                 l.units,
			"unroll":                l.unroll,
			"use_bias":              l.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LcuDNNLSTM) GetCustomLayerDefinition() string {
	return `from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager.context import get_device_name
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import sysconfig
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
import tensorflow.python.framework.config as tfconfig 

# The following string constants are used by Defun approach for unified backend
# of LSTM and GRU.
_FUNCTION_API_NAME_ATTRIBUTE = 'api_implements'
_FUNCTION_DEVICE_ATTRIBUTE = 'api_preferred_device'
_CPU_DEVICE_NAME = 'CPU'
_GPU_DEVICE_NAME = 'GPU'

# The following number constants are used to represent the runtime of the defun
# backend function. Since the CPU/GPU implementation are mathematically same, we
# need some signal for the function to indicate which function is executed. This
# is for testing purpose to verify the correctness of swapping backend function.
_RUNTIME_UNKNOWN = 0
_RUNTIME_CPU = 1
_RUNTIME_GPU = 2

_CUDNN_AVAILABLE_MSG = 'Layer %s will use cuDNN kernels when running on GPU.'
_CUDNN_NOT_AVAILABLE_MSG = ('Layer %s will not use cuDNN kernels since it '
                            'doesn\'t meet the criteria. It will '
                            'use a generic GPU kernel as fallback when running '
                            'on GPU.')

def _runtime(runtime_name):
  with ops.device('/cpu:0'):
    return constant_op.constant(
        runtime_name, dtype=dtypes.float32, name='runtime')

def _use_new_code():
  return False

def _read_variable_value(v):
  """Read the value of a variable if it is variable."""
  if isinstance(v, variables.Variable):
    return v.read_value()
  return v

def _get_context_device_type():
  """Parse the current context and return the device type, eg CPU/GPU."""
  current_device = get_device_name()
  if current_device is None:
    return None
  return device.DeviceSpec.from_string(current_device).device_type

def _canonical_to_params(weights, biases, shape, transpose_weights=False):
  def convert(w):
    return array_ops.transpose(w) if transpose_weights else w

  weights = [array_ops.reshape(convert(x), shape) for x in weights]
  biases = [array_ops.reshape(x, shape) for x in biases]
  return array_ops.concat(weights + biases, axis=0)

def gpu_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias, mask,
             time_major, go_backwards, sequence_lengths):
  if not time_major and mask is None:
    inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
    seq_axis, batch_axis = (0, 1)
  else:
    seq_axis, batch_axis = (0, 1) if time_major else (1, 0)
  # For init_h and init_c, cuDNN expects one more dim of num_layers before or
  # after batch dim for time major or batch major inputs respectively
  init_h = array_ops.expand_dims(init_h, axis=seq_axis)
  init_c = array_ops.expand_dims(init_c, axis=seq_axis)

  weights = array_ops.split(kernel, 4, axis=1)
  weights += array_ops.split(recurrent_kernel, 4, axis=1)
  # CuDNN has an extra set of bias for inputs, we disable them (setting to 0),
  # so that mathematically it is same as the canonical LSTM implementation.
  full_bias = array_ops.concat((array_ops.zeros_like(bias), bias), 0)

  if sysconfig.get_build_info()['is_rocm_build']:
    # ROCm MIOpen's weight sequence for LSTM is different from both canonical
    # and Cudnn format
    # MIOpen: [i, f, o, c] Cudnn/Canonical: [i, f, c, o]
    # i is input gate weights.
    # f is forget gate weights.
    # o is output gate weights.
    # c is cell gate weights.
    weights = [weights[x] for x in (0, 1, 3, 2, 4, 5, 7, 6)]
    # full_bias is a tensor of shape (8*n,)
    full_bias = array_ops.split(full_bias, 8, axis=0)
    full_bias = [full_bias[x] for x in (0, 1, 3, 2, 4, 5, 7, 6)]

  params = _canonical_to_params(
      weights=weights,
      biases=array_ops.split(full_bias, 8),
      shape=constant_op.constant([-1]),
      transpose_weights=True)

  if mask is not None:
    sequence_lengths = calculate_sequence_by_mask(mask, time_major)

  if sequence_lengths is not None:
    if go_backwards:
      # Three reversals are required. E.g.,
      # normal input = [1, 2, 3, 0, 0]  # where 0 need to be masked
      # reversed_input_to_cudnn = [3, 2, 1, 0, 0]
      # output_from_cudnn = [6, 5, 4, 0, 0]
      # expected_output = [0, 0, 6, 5 ,4]
      inputs = array_ops.reverse_sequence_v2(
          inputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
    outputs, h, c, _, _ = gen_cudnn_rnn_ops.CudnnRNNV3(
        input=inputs,
        input_h=init_h,
        input_c=init_c,
        params=params,
        is_training=True,
        rnn_mode='lstm',
        sequence_lengths=sequence_lengths,
        time_major=time_major)
    if go_backwards:
      outputs = array_ops.reverse_sequence_v2(
          outputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
      outputs = array_ops.reverse(outputs, axis=[seq_axis])
  else:
    # # Fill the array with shape [batch] with value of max timesteps.
    # sequence_length = array_ops.fill([array_ops.shape(inputs)[1]],
    #                                  array_ops.shape(inputs)[0])
    if go_backwards:
      # Reverse axis 0 since the input is already convert to time major.
      inputs = array_ops.reverse(inputs, axis=[0])
    outputs, h, c, _ = gen_cudnn_rnn_ops.CudnnRNN(
        input=inputs, input_h=init_h, input_c=init_c, params=params,
        is_training=True, rnn_mode='lstm')

  last_output = outputs[-1]
  if not time_major and mask is None:
    outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
  h = array_ops.squeeze(h, axis=seq_axis)
  c = array_ops.squeeze(c, axis=seq_axis)

  # In the case of variable length input, the cudnn kernel will fill zeros for
  # the output, whereas the default keras behavior is to bring over the previous
  # output for t-1, so that in the return_sequence=False case, user can quickly
  # get the final effect output instead just 0s at the last timestep.
  # In order to mimic the default keras behavior, we copy the final h state as
  # the last_output, since it is numerically same as the output.
  if mask is not None:
    last_output = h
  return last_output, outputs, h, c, _runtime(_RUNTIME_GPU)

def standard_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias,
                  mask, time_major, go_backwards, sequence_lengths,
                  zero_output_for_mask):

  input_shape = backend.int_shape(inputs)
  timesteps = input_shape[0] if time_major else input_shape[1]

  def step(cell_inputs, cell_states):
    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_states[0]  # previous memory state
    c_tm1 = cell_states[1]  # previous carry state

    z = backend.dot(cell_inputs, kernel)
    z += backend.dot(h_tm1, recurrent_kernel)
    z = backend.bias_add(z, bias)

    z0, z1, z2, z3 = array_ops.split(z, 4, axis=1)

    i = nn.sigmoid(z0)
    f = nn.sigmoid(z1)
    c = f * c_tm1 + i * nn.tanh(z2)
    o = nn.sigmoid(z3)

    h = o * nn.tanh(c)
    return h, [h, c]

  last_output, outputs, new_states = backend.rnn(
      step,
      inputs, [init_h, init_c],
      constants=None,
      unroll=False,
      time_major=time_major,
      mask=mask,
      go_backwards=go_backwards,
      input_length=(sequence_lengths
                    if sequence_lengths is not None else timesteps),
      zero_output_for_mask=zero_output_for_mask)
  return (last_output, outputs, new_states[0], new_states[1],
          _runtime(_RUNTIME_CPU))

class GpuLSTM(tf.keras.layers.LSTM):
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               time_major=False,
               unroll=False,
               **kwargs):
    # return_runtime is a flag for testing, which shows the real backend
    # implementation chosen by grappler in graph mode.
    self.return_runtime = kwargs.pop('return_runtime', False)

    super(tf.keras.layers.LSTM, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=kwargs.pop('implementation', 2),
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        time_major=time_major,
        unroll=unroll,
        **kwargs)

    self.state_spec = [
        tf.keras.layers.InputSpec(shape=(None, dim)) for dim in (self.units, self.units)
    ]
    self._could_use_gpu_kernel = (
        self.activation in (tf.keras.activations.tanh, tf.nn.tanh) and
        self.recurrent_activation in (tf.keras.activations.sigmoid, tf.nn.sigmoid) and
        recurrent_dropout == 0 and not unroll and use_bias)

  def call(self, inputs, mask=None, training=None, initial_state=None):
      # The input should be dense, padded with zeros. If a ragged input is fed
      # into the layer, it is padded and the row lengths are used for masking.
      inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
      is_ragged_input = (row_lengths is not None)
      self._validate_args_if_ragged(is_ragged_input, mask)
  
      # LSTM does not support constants. Ignore it during process.
      inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)
  
      if isinstance(mask, list):
        mask = mask[0]
  
      input_shape = backend.int_shape(inputs)
      timesteps = input_shape[0] if self.time_major else input_shape[1]
  
      # TODO(b/156447398) Investigate why the cuDNN kernel fails with ragged
      # inputs.
      if is_ragged_input or not self._could_use_gpu_kernel:
        # Fall back to use the normal LSTM.
        kwargs = {'training': training}
        self._maybe_reset_cell_dropout_mask(self.cell)
  
        def step(inputs, states):
          return self.cell(inputs, states, **kwargs)
  
        last_output, outputs, states = backend.rnn(
            step,
            inputs,
            initial_state,
            constants=None,
            go_backwards=self.go_backwards,
            mask=mask,
            unroll=self.unroll,
            input_length=row_lengths if row_lengths is not None else timesteps,
            time_major=self.time_major,
            zero_output_for_mask=self.zero_output_for_mask)
        runtime = _runtime(_RUNTIME_UNKNOWN)
      else:
        # Use the new defun approach for backend implementation swap.
        # Note that different implementations need to have same function
        # signature, eg, the tensor parameters need to have same shape and dtypes.
        # Since the CuDNN has an extra set of bias, those bias will be passed to
        # both normal and CuDNN implementations.
        self.reset_dropout_mask()
        dropout_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        if dropout_mask is not None:
          inputs = inputs * dropout_mask[0]
        if _use_new_code():
          lstm_kwargs = {
              'inputs':
                  inputs,
              'init_h':
                  _read_variable_value(initial_state[0]),
              'init_c':
                  _read_variable_value(initial_state[1]),
              'kernel':
                  _read_variable_value(self.cell.kernel),
              'recurrent_kernel':
                  _read_variable_value(self.cell.recurrent_kernel),
              'bias':
                  _read_variable_value(self.cell.bias),
              'mask':
                  mask,
              'time_major':
                  self.time_major,
              'go_backwards':
                  self.go_backwards,
              'sequence_lengths':
                  row_lengths,
              'zero_output_for_mask':
                  self.zero_output_for_mask,
          }
          (last_output, outputs, new_h, new_c,
           runtime) = self._defun_wrapper.defun_layer(**lstm_kwargs)
        else:
          gpu_lstm_kwargs = {
              'inputs':
                  inputs,
              'init_h':
                  _read_variable_value(initial_state[0]),
              'init_c':
                  _read_variable_value(initial_state[1]),
              'kernel':
                  _read_variable_value(self.cell.kernel),
              'recurrent_kernel':
                  _read_variable_value(self.cell.recurrent_kernel),
              'bias':
                  _read_variable_value(self.cell.bias),
              'mask':
                  mask,
              'time_major':
                  self.time_major,
              'go_backwards':
                  self.go_backwards,
              'sequence_lengths':
                  row_lengths
          }
          normal_lstm_kwargs = gpu_lstm_kwargs.copy()
          normal_lstm_kwargs.update({
              'zero_output_for_mask': self.zero_output_for_mask,
          })
  
          device_type = _get_context_device_type()
          can_use_gpu = (
              # Either user specified GPU or unspecified but GPU is available.
              (device_type == _GPU_DEVICE_NAME or
               (device_type is None and tfconfig.list_logical_devices('GPU'))) and
              (mask is None or
               is_cudnn_supported_inputs(mask, self.time_major)))
          # Under eager context, check the device placement and prefer the
          # GPU implementation when GPU is available.
          if can_use_gpu:
            last_output, outputs, new_h, new_c, runtime = gpu_lstm(
                **gpu_lstm_kwargs)
          else:
            last_output, outputs, new_h, new_c, runtime = standard_lstm(
                **normal_lstm_kwargs)
  
        states = [new_h, new_c]
  
      if self.stateful:
        updates = [
            state_ops.assign(self_state, state)
            for self_state, state in zip(self.states, states)
        ]
        self.add_update(updates)
  
      if self.return_sequences:
        output = backend.maybe_convert_to_ragged(
            is_ragged_input, outputs, row_lengths, go_backwards=self.go_backwards)
      else:
        output = last_output
  
      if self.return_state:
        return [output] + list(states)
      elif self.return_runtime:
        return output, runtime
      else:
        return output

custom_objects["GpuLSTM"] = GpuLSTM
`
}
