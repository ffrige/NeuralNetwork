Neural Network in Python
========================

This is a simple implementation of a feed-forward neural network in Python. The motivation was to understand the details of back-propagation without relying on external libraries (e.g. Tensorflow).

**FEATURES:**
-	Build custom networks with any layer and units size
-	Sigmoid or ReLU non-linearities
-	Trains with mini-batch gradient descent
-	Gradient check option
-	Rmsprop
-	Dropout (not completed yet)

**TODO:**
-	Fix dropout during backprop
-	Add other loss functions
-	Add softmax non- linearity
-	Add convolutional layers
-	Add more weights initialization options
-	Add other weights regularizations (e.g. L2)


----------


 

Layer Class
-----------

**Layer**(index,units=1,units_prev=1,activation=’relu’)

**Attributes:**

*index* **j**: index of this layer (0..N), where 0 is the input layer and N is the output layer

*units*: number of units in this layer

*units_prev*: number of units in previous layer

*weights* **wj**: matrix of size (units_prev x units), initialized with random small values

*bias* **bj**: vector of size (units), initialized to zero

*activation* **f**: activation function for this layer (ReLU, sigmoid, linear)

*input* **xj**: input values to this layer, that is output of previous layer. Vector of size (units_prev)

*output* **yj**: output values of this layer. Vector of size (units)

*error* **ej**: error of this layer, back-propagated from next layer as ej = ej+1 * wj+1. Error of output layer depends on selected loss function.

*weightGrad* **dwj**: gradients of weights, updated as dwj += ej * f’(xj) * xj / batch_size. The function f’ is the derivative of the activation function for this layer.
biasGrad dbj: gradients of biases, updated as dwj += ej * f’(xj) / batch_size.

*weightCache*: used for rmsprop

*biasCache*: used for rmsprop

*dropoutMask*: used for dropout

*numWeightGrad*: gradients of weights, calculated numerically for gradient check

*numBiasGrad*: gradients of biases, calculated numerically for gradient check

**Private Methods:**

sigmoid(x): sigmoid activation function

sigmoid_derivative(x): derivative of the sigmoid activation function

relu(x): rectified linear unit activation function

relu_derivative(x): derivative of the relu activation function

linear(x): linear activation function

linear_derivative(x): derivative of the linear activation function

activation_function(x): calculate activation function according to layer class attribute “activation”

activation_function_derivative(x): calculate derivative of activation function according to layer class attribute “activation”

**Public Methods:**

*setInput*(input): set the input values xj to this layer.

*getOutput*(dropout=1): calculates the output of this layer. An optional dropout value specifies the probability of keeping the units in the calculation (1=no dropout, 0=all units dropped).

*getWeights*(): returns all the weights wj and biases bj of this layer

*setWeights*(weights, bias): set all weights and biases of this layer

*updateWeights*(learning_rate, gradients, optimizer): update the values of weights and biases according to the given gradients and learning rate. This operation is executed once for each batch. Note that different optimizer use different update formulas: e.g. SGD uses the simple update wj -= learning_rate * gradients.

*getUnits*(): returns the number of units in this layer

*getActivation*(): returns the activation function used in this layer

*setError*(newError): sets a new error value ej for this layer

*getError*(): returns the current error value ej for this layer

*updateWeightsStep*(batch_size): increases the stored gradients for weights and biases according to the current error. This operation is normally executed once for each example in the batch.

*resetWeightsStep*(): resets all internal stored attributes (weightGrad/biasGrad, weightCache/biasCache, numWeightGrad/numBiasGrad). This operation is normally executed at the beginning of each batch iteration.

*getGradients*(): returns the currently stored gradients

*updateNumGradients*(weightGrad, biasGrad, batch_size): increases the stored numerical gradients by the given values: e.g. for weights numWeightGrad += weightGrad / batch_size

*getNumGradients*(): returns the stored numerical gradients for this layer


----------
NeuralNetwork Class
-------------------

**NeuralNetwork**(input_size=1)

**Attributes:**

*input_size*: size of the input value, corresponds to the number of units of the input Layer.

*nb_layers*: number of layers in the network, without counting the input layer. It corresponds to the index of the last layer. It is automatically updated when a new layer is added while building the network.

*Layer*: array of Layer objects. The input layer is automatically added when declaring the network.

**Private Methods:**

*lossFunction*(input_values, target_values, metrics="mse"): evaluates the current loss of the network given a set of input and target values. The mean square error is used as default metric.

*numGrad*(X, y, batch_size): calculates the numerical gradients of the loss function w.r.t. all weights and biases

**Public Methods:**

*printModel*(): prints basic information about each layer in the network
addLayer(units, activation): appends a new layer at the end of the network, with the specified number of units and the given activation function.

*printWeights*(): prints out all the weights and biases of the network ordered by layer.

*loadWeights*(filename): loads all the weights and biases of the network from the specified file (TODO)

*saveWeights*(filename): saves all the weights and biases of the network to the specified file (TODO)

*predict*(inputs): calculates the network’s output values given the specified input values.

*train*(training_set_inputs, training_set_targets, learning_rate, optimizer="rmsprop", metric="mse", validation_size=0.2, dropout=1, epochs=1, batch_size=1, logger=False, plot=False, gradCheck=False): trains the weights of the network according to the given input and target values.

*setScaler*(mean, variance): sets the mean and variance to scale the input values (TODO)


----------
Example of use:
---------------

    import numpy as np
    import matplotlib.pyplot as plt
    from NN import NeuralNetwork
    
    #build network with one input value, one hidden layer of 20 units, and one output value
    model = NeuralNetwork(input_size=1)
    model.addLayer(units=20,activation='relu')
    model.addLayer(units=1,activation='linear')
     
    #train network
    model.train(training_set_inputs,training_set_targets,learning_rate=0.01,epochs=1000,batch_size=64,
                dropout=0.6,validation_size=0.2,optimizer='sgd',logger=False,plot=True,gradCheck=False)
    
    #show results
    model.predict([0.2])




