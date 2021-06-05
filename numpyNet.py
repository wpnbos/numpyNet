import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
from timeit import timeit
from nnfs.datasets import sine_data

nnfs.init()

np.random.seed(0)

class layer_input:
	def forward(self, inputs):
		self.output = inputs

class layer_dense:
	def __init__(self, n_inputs, n_neurons, L1_w=0, L1_b=0, L2_w=0, L2_b=0):
		self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
		self.L1_w = L1_w
		self.L1_b = L1_b
		self.L2_w = L2_w
		self.L2_b = L2_b
		
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases
		self.inputs = inputs

	def backward(self, dvalues):
		self.d_weights = np.dot(self.inputs.T, dvalues)
		self.d_biases = np.sum(dvalues, axis=0, keepdims=True)
		self.d_inputs = np.dot(dvalues, self.weights.T)

		if self.L1_w > 0:
			dL1 = np.ones_like(self.weights)
			dL1[self.weights < 0] = -1
			self.dweights += self.L1_w * dL1

		if self.L2_w > 0:
			self.d_weights += 2 * self.L2_w * self.weights

		if self.L1_b > 0:
			dL1 = np.ones_like(self.biases)
			dL1[self.biases < 0] = -1
			self.d_biases += self.L1_b * dL1

		if self.L2_b > 0:
			self.d_biases += 2 * self.L2_b * self.biases

class layer_dropout():
	def __init__(self, rate):
		self.rate = 1 - rate

	def forward(self, inputs):
		self.inputs = inputs
		self.dropout_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
		#apply mask
		self.output = inputs * self.dropout_mask

	def backward(self, dvalues):
		self.d_inputs = dvalues * self.dropout_mask

class activation_ReLU: 
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)
		self.inputs = inputs

	def backward(self, dvalues):
		self.d_inputs = dvalues.copy()
		self.d_inputs[self.inputs <= 0] = 0

class activation_softmax:
	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities

	def backward(self, dvalues):
		self.d_inputs = np.empty_like(dvalues)
		for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
			single_output = single_output.reshape(-1, 1)
			jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
			self.d_inputs[index] = np.dot(jacobian_matrix, single_dvalues)

class activation_sigmoid():
	def forward(self, inputs):
		self.inputs = inputs
		self.output = 1 / (1 + np.exp(-inputs))

	def backward(self, dvalues):
		self.d_inputs = dvalues * (1 - self.output) * self.output
			
class activation_linear:
	def forward(self, inputs):
		self.inputs = inputs
		self.output = inputs

	def backward(self, dvalues):
		self.d_inputs = dvalues.copy()

class loss():
	def calculate(self, output, y):
		sample_losses = self.forward(output, y)
		data_loss = np.mean(sample_losses)
		return data_loss
	def regularization_loss(self, layer):
		regularization_loss = 0

		if layer.L1_w > 0: 
			regularization_loss += layer.L1_w * np.sum(np.abs(layer.weights))
		if layer.L1_b > 0: 
			regularization_loss += layer.L1_b * np.sum(np.abs(layer.biases))
		if layer.L2_w > 0: 
			regularization_loss += layer.L2_w * np.sum(np.abs(layer.weights))			
		if layer.L2_b > 0: 
			regularization_loss += layer.L2_b * np.sum(np.abs(layer.biases))

		return regularization_loss

class loss_meansquarederror(loss):
	def forward(self, y_pred, y_true):
		sample_losses = np.mean((y_true - y_pred)**2, axis = -1)
		return sample_losses

	def backward(self, y_pred, y_true):
		samples = len(y_pred)
		outputs = len(y_pred[0])
		self.d_inputs = -2 * (y_true - y_pred) / outputs
		self.d_inputs = self.d_inputs / samples 

class loss_categorical_crossEntropy(loss):
	def forward(self, y_pred, y_true):
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
		if len(y_true.shape) == 1:
			confidences = y_pred_clipped[range(samples), y_true]

		elif len(y_true.shape) == 2:
			confidences = np.sum(y_pred_clipped * y_true, axis = 1)

		negative_log_likelihoods = -np.log(confidences)
		return negative_log_likelihoods

	def backward(self, y_pred, y_true):
		samples = len(y_pred)
		labels = len(y_pred[0])
		if len(y_true.shape) == 1:
			y_true = np.eye(labels)[y_true]
		self.d_inputs = -(y_true / y_pred)
		self.d_inputs = self.d_inputs / samples

class loss_binarycrossentropy(loss):
	def forward(self, y_pred, y_true):
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
		sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
		sample_losses = np.mean(sample_losses, axis = -1)
		return sample_losses

	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		outputs = len(dvalues[0])
		clipped_d_values = np.clip(dvalues, 1e-7, 1 - 1e-7)
		self.d_inputs = -(y_true / clipped_d_values - (1 - y_true) / (1 - clipped_d_values)) / outputs
		self.d_inputs = self.d_inputs / samples 

class activation_softmax_loss_categorical_crossEntropy():
	def __init__(self):
		self.activation = activation_softmax()
		self.loss = loss_categorical_crossEntropy()

	def forward(self, inputs, y_true):
		self.activation.forward(inputs)
		self.output = self.activation.output
		return self.loss.calculate(self.output, y_true)

	def backward(self, y_pred, y_true):
		#get number of samples
		samples = len(y_pred)
		#if y_true has more than one row, it's one hot encoded
		#transform it to sparse values
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis=1)

		self.d_inputs = y_pred.copy()
		#calculate derivative d/dx = y_pred - y_true (y_true is always 1)
		self.d_inputs[range(samples), y_true] -= 1
		#normalize the derivative
		self.d_inputs = self.d_inputs / samples 

class optimizer_sgd:
	#initialize object, save settings 
	def __init__(self, learning_rate=1.0, decay = 0., momentum = 0.):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.momentum = momentum 

	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

	#update parameters of given layer
	def update_params(self, layer):
		if self.momentum:
			if not hasattr(layer, 'weight_momentums'):
				layer.weight_momentums = np.zeros_like(layer.weights)
				layer.bias_momentums = np.zeros_like(layer.biases)

			weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.d_weights
			layer.weight_momentums = weight_updates
			bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.d_biases
			layer.bias_momentums = bias_updates
		else: 
			weight_updates = self.current_learning_rate * layer.d_weights
			bias_updates = self.current_learning_rate * layer.d_biases

		layer.weights += weight_updates
		layer.biases += bias_updates


	def post_update_params(self):
		self.iterations += 1

class optimizer_adagrad:
	#initialize object, save settings 
	def __init__(self, learning_rate=1.0, decay = 0., epsilon = 1e-7):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon

	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

	#update parameters of given layer
	def update_params(self, layer):
		#if weight and bias cache not present, initialize them
		if not hasattr(layer, 'weight_cache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_cache = np.zeros_like(layer.biases)

		#update cache by adding this rounds' derivatives squared
		layer.weight_cache += layer.d_weights**2
		layer.bias_cache += layer.d_biases**2
		
		layer.weights += -self.current_learning_rate * layer.d_weights / (np.sqrt(layer.weight_cache) + self.epsilon)
		layer.biases += -self.current_learning_rate * layer.d_biases / (np.sqrt(layer.bias_cache) + self.epsilon)

	def post_update_params(self):
		self.iterations += 1

class optimizer_rprop:
	#initialize object, save settings 
	def __init__(self, learning_rate=1.0, delta = 0.0125, alpha=1.2, beta=0.5, nmin=1e-6, nmax=50, iRprop=False):
		self.current_learning_rate = learning_rate
		self.alpha = alpha
		self.beta = beta
		self.nmin = nmin
		self.nmax = nmax
		self.iterations = 0
		self.iRprop = iRprop
		self.delta = delta

	def pre_update_params(self):
		yoodledoot = 1

	#update parameters of given layer
	def update_params(self, layer, loss=None):
		#initialize caches of weights and biases
		if not hasattr(layer, 'nweights'):
			layer.nweights = np.full(layer.weights.shape, self.delta)
			layer.nbiases = np.full(layer.biases.shape, self.delta)
			layer.d_weightsOld = layer.d_weights
			layer.d_biasesOld = layer.d_biases

		#calculate combined signs of current and previous layer
		layer.combsign_weights = layer.d_weights * layer.d_weightsOld
		layer.combsign_biases = layer.d_biases * layer.d_biasesOld

		#update step size n
		new_nweights = np.where(layer.combsign_weights > 0, np.minimum(layer.nweights * self.alpha, self.nmax), np.maximum(layer.nweights * self.beta, self.nmin))
		new_nweights = np.where(layer.combsign_weights == 0, layer.nweights, new_nweights)
		new_nbiases = np.where(layer.combsign_biases > 0, np.minimum(layer.nbiases * self.alpha, self.nmax), np.maximum(layer.nbiases * self.beta, self.nmin))
		new_nbiases = np.where(layer.combsign_biases == 0, layer.nbiases, new_nbiases)

		if self.iRprop == True:
			layer.d_weights = np.where(layer.combsign_weights < 0, 0, layer.d_weights)
			layer.d_biases = np.where(layer.combsign_biases < 0, 0, layer.d_biases)

		#calculate new layer weights
		layer.weights = layer.weights - new_nweights * np.sign(layer.d_weightsOld)
		layer.biases = layer.biases - new_nbiases * np.sign(layer.d_biasesOld)

		#store new values to be used in next iteration
		layer.nweights = new_nweights
		layer.nbiases = new_nbiases
		layer.d_biasesOld = layer.d_biases
		layer.d_weightsOld = layer.d_weights

	def post_update_params(self):
		self.iterations += 1

class optimizer_rmsprop:
	def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon
		self.rho = rho

	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

	def update_params(self, layer):
		if not hasattr(layer, 'weight_cache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_cache = np.zeros_like(layer.biases)

		layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.d_weights**2
		layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.d_biases**2

		layer.weights += -self.current_learning_rate * layer.d_weights / (np.sqrt(layer.weight_cache) + self.epsilon)
		layer.biases += -self.current_learning_rate * layer.d_biases / (np.sqrt(layer.bias_cache) + self.epsilon)

	def post_update_params(self):
		self.iterations += 1

class optimizer_adam:
	def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon
		self.beta_1 = beta_1
		self.beta_2 = beta_2

	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

	def update_params(self, layer):
		if not hasattr(layer, 'weight_cache'):
			layer.weight_momentums = np.zeros_like(layer.weights)
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_momentums = np.zeros_like(layer.biases)
			layer.bias_cache = np.zeros_like(layer.biases)

		layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.d_weights
		layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.d_biases

		#corrected momentum
		weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
		bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
		#update cache with squared current gradients
		layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.d_weights**2
		layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.d_biases**2

		weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
		bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        #parameter update with normalization and square rooted cache
		layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
		layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

	def post_update_params(self):
		self.iterations += 1

class model:
	def __init__(self):
		self.layers = []

	def add(self, layer):
		self.layers.append(layer)

	def set(self, *, loss, optimizer):
		self.loss = loss
		self.optimizer = optimizer 

	def train(self, X, y, *, epochs = 1, print_every=1):
		for epoch in range(1, epochs+1):
			pass


X, y = sine_data()

model = model()

model.add(layer_dense(1, 64))
model.add(activation_ReLU())
model.add(layer_dense(64, 64))
model.add(activation_ReLU())
model.add(layer_dense(64, 1))
model.add(activation_linear())

model.set(loss = loss_meansquarederror(), optimizer=optimizer_adam(learning_rate=0.005, decay=1e-3))

model.train(X, y, epochs = 10000, print_every=100)