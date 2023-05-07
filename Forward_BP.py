# -*- coding: utf-8 -*-
"""
Created on Sat Apr  15 12:32:21 2023

@author: premchand
"""

import pandas as pd
import random
import math
import numpy as np
import sys
import logging
LOG_FILENAME = 'example.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

# network initialization
def networkInitialization(numberOfInputs, numberOfOutputs, numberOfHiddenLayers, numberOfNeuronsPerLayer):
    weightList = list()
    for index in range(numberOfHiddenLayers + 1):
        if index == 0:
            hiddenLayer = [[random.random() for i in range(numberOfInputs + 1)] for i in
                           range(numberOfNeuronsPerLayer[index])]
        elif index == numberOfHiddenLayers:
            hiddenLayer = [[random.random() for i in range(numberOfNeuronsPerLayer[index - 1] + 1)] for i in
                           range(numberOfOutputs)]
        else:
            hiddenLayer = [[random.random() for i in range(numberOfNeuronsPerLayer[index - 1] + 1)] for i in
                           range(numberOfNeuronsPerLayer[index])]
        weightList.append(hiddenLayer)
    return weightList


# activation
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return transfer(activation)


# Sigmoid Function
def transfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))


# Forward Propagation
def forwardPropagate(weightList, dataRow):
    inputs = dataRow[:-1]
    eachNeuronOutputs = list()
    eachNeuronOutputs.append(dataRow[:-1])
    for eachLayerIndex in range(len(weightList)):
        newInputs = []
        for eachNeuronIndex in range(len(weightList[eachLayerIndex])):
            activation = activate(weightList[eachLayerIndex][eachNeuronIndex], inputs)
            newInputs.append(activation)
        inputs = newInputs
        eachNeuronOutputs.append(newInputs)
    return eachNeuronOutputs


# BackWard Propagate
def backwardPropagate(outputs, expectedOutput, weightList, learningRate):
    # expectedOutput = [0,1]
    deltaList = []
    for indexOutputLayer in reversed(range(len(outputs))):
        newDelta = list()
        if indexOutputLayer == len(outputs) - 1:
            for index in range(len(outputs[indexOutputLayer])):
                delta = outputs[indexOutputLayer][index] * (1 - outputs[indexOutputLayer][index]) * (
                    expectedOutput[index] - outputs[indexOutputLayer][index])
                newDelta.append(delta)
            deltaList = newDelta
        elif indexOutputLayer == 0:
            for index in range(len(outputs[indexOutputLayer])):
                for indexWeight in range(len(weightList[indexOutputLayer])):
                    weightList[indexOutputLayer][indexWeight][index] += learningRate * deltaList[indexWeight] * \
                                                                       outputs[indexOutputLayer][index]
            for indexWeight in range(len(weightList[indexOutputLayer])):
                weightList[indexOutputLayer][indexWeight][-1] += learningRate * deltaList[indexWeight]
        else:
            for index in range(len(outputs[indexOutputLayer])):
                sum = 0
                for indexWeight in range(len(weightList[indexOutputLayer])):
                    sum += weightList[indexOutputLayer][indexWeight][index] * deltaList[indexWeight]
                    weightList[indexOutputLayer][indexWeight][index] += learningRate * deltaList[indexWeight] * \
                                                                        outputs[indexOutputLayer][index]
                delta = outputs[indexOutputLayer][index] * (1 - outputs[indexOutputLayer][index]) * sum
                newDelta.append(delta)
            for indexWeight in range(len(weightList[indexOutputLayer])):
                weightList[indexOutputLayer][indexWeight][-1] += learningRate * deltaList[indexWeight]
            deltaList = newDelta


def train_network(weightList, traininigDataSet, learningRate, noOfIteration, numberOfOutputs):
    for iter in range(noOfIteration):
        sum_error = 0
        for row in traininigDataSet:
            outputs = forwardPropagate(weightList, row)
            expected = [0 for i in range(numberOfOutputs)]
            expected[int(row[-1]) - 1] = 1
            #actuals = maxOutput(outputs[len(outputs) - 1])
            # sum_error += (sum([(expected[i] - actuals[i]) ** 2 for i in range(len(expected))]) / 2)
            sum_error += sum([(expected[i] - outputs[len(outputs) - 1][i]) ** 2 for i in range(len(expected))])
            backwardPropagate(outputs,expected,weightList,learningRate)
        sum_error = sum_error/len(traininigDataSet)
        print('Iteration=%d, Error=%.8f' % (iter+1,  sum_error))
        precisedError = '%.8f' % sum_error
        if float(precisedError) == 0.0:
            break

def printWeights(weights):
    for layer in range(len(weights)):
        print("Layer " + str(layer) + ":")
        for col in range(len(weights[layer][0])):
            neuronWeights = []
            for row in range(len(weights[layer])):
                neuronWeights.append(weights[layer][row][col])
            if(col == len(weights[layer][0])-1):
                print("\t Bias Term :" + str(neuronWeights))
            else:
                print("\t Neuron " + str(col+1)+ " : " + str(neuronWeights))

def maxOutput(Lastoutputs):
    actuals = [0 for i in range(len(Lastoutputs))]
    index = Lastoutputs.index(max(Lastoutputs))
    actuals[index] = 1;
    return actuals

def testTheModel(dataSet, weightList,numberOfOutputs):
    sumError = 0
    count = 0
    for data in dataSet:
        outputs = forwardPropagate(weightList,data)
        expected = [0 for i in range(numberOfOutputs)]
        expected[int(data[-1]) - 1] = 1
        sumError += sum([(expected[i] - outputs[len(outputs) - 1][i]) ** 2 for i in range(len(expected))])
        actuals = maxOutput(outputs[len(outputs) - 1])
        if actuals == expected:
            count = count + 1
    sumError = sumError / len(dataSet)
    accuracy = count/len(dataSet)
    outs = []
    outs.append(sumError)
    outs.append(accuracy)
    return outs





