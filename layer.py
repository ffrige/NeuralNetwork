import numpy as np

class Layer:
    "Individual layer of a neural network"

    def __init__(self,index,units=1,units_prev=1,activation='relu'):

        # attributes
        self.index = index
        self.units = units
        self.units_prev = units_prev
        self.bias = np.zeros(self.units)
        self.weights = np.random.normal(0.0, 2.0/(self.units_prev+self.units),(self.units_prev,self.units)) #TODO - give options for initialization (uniform, truncated normal)
        self.activation = activation
        self.input = np.zeros(self.units_prev)
        self.output = np.zeros(self.units)
        self.error = np.zeros(self.units)
        self.weightGrad = np.zeros([self.units_prev,self.units])
        self.biasGrad = np.zeros([self.units])
        self.weightCache = self.weightGrad # for RMSprop
        self.biasCache = self.biasGrad # for RMSprop
        self.dropoutMask = np.zeros([self.units_prev,self.units])
        self.numWeightGrad = np.zeros([self.units_prev,self.units])
        self.numBiasGrad = np.zeros([self.units])

    """
    PRIVATE METHODS
    """
    
    def __sigmoid(self,x):
        return 1.0 / (1.0+np.exp(-x)) 

    def __sigmoid_derivative(self,x):
        y = self.__sigmoid(x)
        return y * (1.0-y)
    
    def __relu(self,x):
        x[x<0] = 0
        return x

    def __relu_derivative(self,x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def __linear(self,x):
        return x

    def __linear_derivative(self,x):
        return np.ones_like(x)

    def __softmax(self,x):
        #Compute the softmax of vector x in a numerically stable way
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)

    def __softmax_derivative(self,x):
        Sz = self.__softmax(x)
        D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
        return D

    def __activation_function(self,x):
        if self.activation == "sigmoid":
            return self.__sigmoid(x)
        elif self.activation == "relu":
            return self.__relu(x)
        elif self.activation == "linear":
            return self.__linear(x)
        elif self.activation == "softmax":
            return self.__softmax(x)
        else:
            assert False,"Selected activation function does not exist!"

    def __activation_function_derivative(self,x):
        if self.activation == "sigmoid":
            return self.__sigmoid_derivative(x)
        elif self.activation == "relu":
            return self.__relu_derivative(x)
        elif self.activation == "linear":
            return self.__linear_derivative(x)
        elif self.activation == "softmax":
            return self.__softmax_derivative(x)
        else:
            assert False,"Selected activation function does not exist!"


    """
    PUBLIC METHODS
    """

    def setInput(self,input):
        assert (len(self.input) == len(input)),"Wrong layer input size!"
        self.input = input

    def getOutput(self,dropout=1):
        self.output = self.__activation_function(np.dot(self.input,self.weights)+self.bias)
        self.dropoutMask = (np.random.rand(*self.output.shape) <= dropout) / dropout
        self.output *= self.dropoutMask 
        return self.output

    def getWeights(self):
        return self.weights,self.bias

    def setWeights(self,weights,bias):
        assert ((len(self.weights) == len(weights))and(len(self.bias) == len(bias))),"Wrong weights/bias size!"
        self.weights = weights
        self.bias = bias                

    def updateWeights(self,learning_rate,gradients,optimizer):
        #return gradient
        if optimizer == "sgd":
            self.weights += learning_rate * gradients[0]
            self.bias += learning_rate * gradients[1]
        elif optimizer == "rmsprop":
            eps = 1e-7 # prevents division by zero in rmsprop
            self.weights += learning_rate *  gradients[0] / (np.sqrt(self.weightCache) + eps)
            self.bias += learning_rate * gradients[1] / (np.sqrt(self.biasCache) + eps)
        else:
            assert False,"Selected optimizer does not exist!"
        
    def getUnits(self):
        return self.units

    def getActivation(self):
        return self.activation

    def setError(self,newError):
        self.error = newError

    def getError(self):
        return self.error
    
    def updateWeightsStep(self,batch_size):
        #TODO: add regularization here!
        """ TEST - TEST - TEST
        z = np.dot(self.input,self.weights)+self.bias
        print("z shape",z.shape)
        print("Error shape",self.error.shape)
        aaa = self.__activation_function_derivative(z)
        bbb = self.error*aaa
        print("ActivationFunctionDerivative shape",aaa.shape)
        print("ErrorxAFD shape",bbb.shape)
        print("Input shape",self.input.reshape(self.units_prev,1).shape)
        self.weightGrad += np.dot(self.input.reshape(self.units_prev,1),bbb.reshape(1,self.units)) / batch_size
        print("Gradient shape",self.weightGrad.shape)
        """
        self.weightGrad += self.error * self.__activation_function_derivative(np.dot(self.input,self.weights)+self.bias) * self.input.reshape(self.units_prev,1) / batch_size
        #TODO: add dropoutMask here!
        #self.weightGrad[self.dropoutMask>0] += ...
        self.biasGrad += self.error * self.__activation_function_derivative(np.dot(self.input,self.weights)+self.bias) / batch_size
        self.weightCache = 0.9 * self.weightCache + (1 - 0.9) * self.weightGrad**2
        #self.weightCache[self.dropoutMask>0] = ...
        self.biasCache = 0.9 * self.biasCache + (1 - 0.9) * self.biasGrad**2
    
    def resetWeightsStep(self):
        self.weightGrad = np.zeros([self.units_prev,self.units])
        self.biasGrad = np.zeros([self.units])
        self.weightCache = np.zeros([self.units_prev,self.units])
        self.biasCache = np.zeros([self.units])
        self.numWeightGrad = np.zeros([self.units_prev,self.units])
        self.numBiasGrad = np.zeros([self.units])

    def getGradients(self):
        return self.weightGrad, self.biasGrad

    def updateNumGradients(self,weightGrad,biasGrad,batch_size):
        self.numWeightGrad += weightGrad / batch_size
        self.numBiasGrad += biasGrad / batch_size

    def getNumGradients(self):
        return self.numWeightGrad,self.numBiasGrad
