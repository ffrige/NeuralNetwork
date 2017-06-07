import numpy as np
import matplotlib.pyplot as plt
from layer import Layer

class NeuralNetwork:
    "Implements a feed-forward neural network"

    def __init__(self,input_size=1):

        # hyperparameters
        self.input_size = input_size
        self.nb_layers = 0

        self.Layer = []
        self.Layer.append(Layer(0,self.input_size))


    """
    PRIVATE METHODS
    """

    def __lossFunction(self,predicted_values,target_values,cost):
        tmpLoss = 0

        #extract number of features and examples
        nb_features = self.Layer[self.nb_layers].getUnits()
        nb_examples = predicted_values.shape[0]

        #no loss if no training example provided
        if nb_examples < 1:
            return 0
        
        if cost=="mse":  #Mean Square Error
            #go through each training example
            for p, y in zip(predicted_values, target_values):
                tmpLoss += np.sum((y-p)**2)/nb_features
            return tmpLoss/nb_examples

        elif cost=="log": #Log Likelihood -> should be used with softmax layer
            #go through each training example
            for p, y in zip(predicted_values, target_values):
                tmpLoss -= np.sum(y*np.log(p))/nb_features
            return tmpLoss/nb_examples
            
        elif cost=="ce": #Cross-Entropy -> should be used with sigmoid layer
            #go through each training example
            for p, y in zip(predicted_values, target_values):
                tmpLoss -= np.sum(y*np.log(p)+(1-y)*np.log(1-p))/nb_features
            return tmpLoss/nb_examples
            
        else:
            assert False,"Unknown loss function!"

    def __lossFunctionDerivative(self,predicted_values,target_values,cost):
        
        if cost=="mse":  #Mean Square Error
            return (target_values-predicted_values)

        elif cost=="log": #Log Likelihood
            hot_encoded = np.zeros_like(predicted_values)
            py = predicted_values[np.argmax(target_values)]
            hot_encoded[np.argmax(target_values)] = -1/py.flat[0]
            return hot_encoded.flatten()
            
        elif cost=="ce": #Cross-Entropy
            return (target_values-predicted_values)
        
        else:
            assert False,"Unknown loss function!"


    
    def __numGrad(self,X,y,batch_size,cost):
        delta = 1e-6
        for i in range(self.nb_layers,0,-1):
            oldWeights, oldBiases = self.Layer[i].getWeights()
            tmpWeights = oldWeights
            tmpBiases = oldBiases
            numWeightsGradient = np.empty(tmpWeights.shape)
            numBiasesGradient = np.empty(tmpBiases.shape)
            for tmpWeight,tmpWeightGrad in np.nditer([tmpWeights,numWeightsGradient],op_flags=['readwrite']):
                tmpWeight[...] += delta
                self.Layer[i].setWeights(tmpWeights,oldBiases)
                y_plus = self.predict(X)
                tmpWeights = oldWeights
                tmpWeight[...] -= delta
                self.Layer[i].setWeights(tmpWeights,oldBiases)
                y_minus = self.predict(X)
                tmpWeightGrad[...] = (self.__lossFunction(y_minus,y,cost)-self.__lossFunction(y_plus,y,cost))/(2*delta)
                #restore old weights
                self.Layer[i].setWeights(oldWeights,oldBiases)
            for tmpBias,tmpBiasGrad in np.nditer([tmpBiases,numBiasesGradient],op_flags=['readwrite']):
                tmpBias[...] += delta
                self.Layer[i].setWeights(oldWeights,tmpBiases)
                y_plus = self.predict(X)
                tmpBiases = oldBiases
                tmpBias[...] -= delta
                self.Layer[i].setWeights(oldWeights,tmpBiases)
                y_minus = self.predict(X)
                tmpBiasGrad[...] = (self.__lossFunction(y_minus,y,cost)-self.__lossFunction(y_plus,y,cost))/(2*delta)
                #restore old weights
                self.Layer[i].setWeights(oldWeights,oldBiases)
            self.Layer[i].updateNumGradients(numWeightsGradient,numBiasesGradient,batch_size)

        

    """
    PUBLIC METHODS
    """

    def printModel(self):
        print("0: INPUT layer with {0} units".format(self.input_size))
        if self.nb_layers == 0:
            return
        for i in range(self.nb_layers-1):
            print("{3}: HIDDEN layer {0} with {1} units and {2} activation function".format(i+1,self.Layer[i+1].getUnits(),self.Layer[i+1].getActivation(),i+1))
        print("{2}: OUTPUT layer with {0} units and {1} activation function".format(self.Layer[self.nb_layers].getUnits(),self.Layer[self.nb_layers].getActivation(),self.nb_layers))

    def addLayer(self,units,activation): #TODO - add weight initialization option
        self.nb_layers += 1
        units_prev = self.Layer[self.nb_layers-1].getUnits()
        self.Layer.append(Layer(self.nb_layers,units,units_prev,activation))

    def printWeights(self):
        for i in range(self.nb_layers):
            print("Layer {0} to {1}:\n".format(i,i+1),self.Layer[i+1].getWeights())

    def loadWeights(self,fileName):
        pass

    def saveWeights(self,fileName):
        pass
    
    def predict(self,inputs):
	#check input size
        assert (len(inputs)==self.input_size), "Input size is not correct! Expected %d, given %d" %(self.input_size,len(inputs))
        assert (self.nb_layers > 0), "No output layer defined!"
        
        self.Layer[1].setInput(inputs)
        
        for i in range(1,self.nb_layers):
            self.Layer[i+1].setInput(self.Layer[i].getOutput())

        return self.Layer[self.nb_layers].getOutput()
    
    def train(self,training_set_inputs,training_set_targets,learning_rate,optimizer="sgd",cost="mse",
              validation_size=0,dropout=1,epochs=1,batch_size=1,logger=False,plot=False,gradCheck=False):

        #check parameters
        assert(training_set_inputs.shape[1] == self.input_size), "Size of training input values is not correct! Expected %d, given %d" % (self.input_size, training_set_inputs.shape[1])
        assert(training_set_targets.shape[1] == self.Layer[self.nb_layers].getUnits()), "Size of training target values is not correct! Expected %d, given %d" % (self.Layer[self.nb_layers].getUnits(), training_set_targets.shape[1])
        assert(training_set_inputs.shape[0] == training_set_targets.shape[0]), "Size of input and target values must be equal!"
        assert(self.nb_layers > 0), "No output layer defined!"
        assert(epochs > 0), "At least 1 epoch required!"
        assert(validation_size < 1), "Validation size too large!"

        train_loss = []
        validation_loss = []
        grad_diffWeight = []
        grad_diffBias = []
        grad_Weight = []
        grad_Bias = []
        
        n_examples = training_set_inputs.shape[0]

        #extract and remove cross-validation data from training set
        validation = np.random.choice(n_examples,int(validation_size*n_examples))
        validation_inputs, validation_targets = training_set_inputs[validation], training_set_targets[validation]
        validation_predictions = np.zeros_like(validation_targets)
        training_set_inputs = np.delete(training_set_inputs, validation, 0)
        training_set_targets = np.delete(training_set_targets, validation, 0)
        training_set_predictions = np.zeros_like(training_set_targets)
        n_examples = training_set_inputs.shape[0]
        assert(batch_size<=n_examples), "Batch size larger than (training set - validation)!"

        #clear cache for rms
        for i in range(1,self.nb_layers):
            self.Layer[i].resetCache()
      
        #repeat for all epochs
        for epoch in range(epochs):

            #make a local copy of training set
            tmp_training_set_inputs = training_set_inputs
            tmp_training_set_targets = training_set_targets

            #repeat for all batches
            for batch_index in range(int(n_examples/batch_size)):

                n_examples_left = tmp_training_set_inputs.shape[0]
                
                #extract random batch_size
                batch = np.random.choice(n_examples_left,batch_size)
                
                batch_inputs, batch_targets = tmp_training_set_inputs[batch], tmp_training_set_targets[batch]
                tmp_training_set_inputs = np.delete(tmp_training_set_inputs, batch, 0)
                tmp_training_set_targets = np.delete(tmp_training_set_targets, batch, 0)


                #clear old values
                for i in range(1,self.nb_layers):
                    self.Layer[i].resetGradients()

                #repeat for each training example
                for X, y in zip(batch_inputs, batch_targets):

                    #forward pass
                    self.Layer[1].setInput(X)
                    for i in range(1,self.nb_layers):
                        self.Layer[i+1].setInput(self.Layer[i].getOutput(dropout))
                    
                    #calculate error
                    output_error = self.__lossFunctionDerivative(self.Layer[self.nb_layers].getOutput(),y,cost)                    

                    #backward pass
                    for i in range(self.nb_layers,0,-1):
                        if i == self.nb_layers:
                            self.Layer[i].setError(output_error)
                        else:
                            #backpropagate error
                            self.Layer[i].setError(np.dot(self.Layer[i+1].getWeights()[0],self.Layer[i+1].getError()))
                        self.Layer[i].updateGradients(batch_size)

                    #calculate numerical gradients for gradient check
                    if gradCheck:
                        self.__numGrad(X,y,batch_size,cost)

                #update weights after each batch
                for i in range(self.nb_layers,0,-1):
                    gradients = self.Layer[i].getGradients()
                    self.Layer[i].updateWeights(learning_rate,gradients,optimizer)

                    #gradient check (optional)
                    if gradCheck:
                        numGradients = self.Layer[i].getNumGradients()
                        #calculate relative error of gradients
                        #avoid dividing by zero -> zero gradients are ok (e.g. ReLU)
                        gradsum = np.linalg.norm(gradients[0])+np.linalg.norm(numGradients[0])
                        if gradsum == 0:
                            weightGradDiff = 0
                            biasGradDiff = 0
                        else:
                            weightGradDiff = np.linalg.norm(gradients[0]-numGradients[0]) / gradsum
                            biasGradDiff = np.linalg.norm(gradients[1]-numGradients[1]) / gradsum
                        grad_diffWeight.append(np.sum(weightGradDiff))
                        grad_diffBias.append(np.sum(biasGradDiff))
                        if (weightGradDiff>1e-4).any() or (biasGradDiff>1e-4).any():
                            print("Gradient error in epoch {0}, Layer {1}!".format(epoch,i))
                            print("weightGradDiff",weightGradDiff)
                            print("biasGradDiff",biasGradDiff)
                            assert False,"Learning aborted!"                    

            #update losses at the end of each epoch
            for i in range(training_set_inputs.shape[0]):
                training_set_predictions[i] = self.predict(training_set_inputs[i])
            for i in range(validation_inputs.shape[0]):
                validation_predictions[i] = self.predict(validation_inputs[i])
            train_loss.append(self.__lossFunction(training_set_predictions,training_set_targets,cost))
            validation_loss.append(self.__lossFunction(validation_predictions,validation_targets,cost))
            grad_Weight.append(np.mean(gradients[0]))
            grad_Bias.append(np.mean(gradients[1]))


            if logger or epoch+1 == epochs:
                print('Epoch: {0}/{1} ... Training loss: {2} ... Validation loss: {3}'.format(epoch+1,epochs,train_loss[epoch],validation_loss[epoch]))

        #plot graph at end of training
        if plot:
            plt.plot(train_loss)
            plt.plot(validation_loss)
            plt.show()
   
    def setScaler(self,mean=0,variance=1):
        self.ScalerMean = mean
        self.ScalerVariance = variance

