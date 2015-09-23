/*****************************************************************
Neural network trainer
Written by Matthieu Courbariaux in 2014

Specifications of the trainer :
- Early stopping
- exponential decay learning rate
- Saturating momentum
*******************************************************************/

#ifndef NNTrain
#define NNTrain

#include "NN.h"
#include "dataset.h"

// Train the NN using gradient descent
void trainNetwork(neuralNetwork* NN, dataSet* trainingSet, dataSet* validationSet, dataSet* testSet, 
	double learningRate, double LR_decay, double LR_min, 
	int earlyStop, 
	double momentum, int momentum_sat, double momentum_fin);

#endif


