
#include <iostream>
#include "dataset.h"
#include "train.h"
#include <Eigen/Core>
using namespace std;

int main(int argc, char* argv[])
{
	cout << endl << "-- BEGINNING OF PROGRAM --" << endl<<endl;	
	
	// set the number of threads for openmp
	// I have 12 cores on 2 sockets
	// 12 cores -> 30 secondes per epoch
	// 2 cores -> 35 secondes per epoch
	// 1 core -> 45 secondes per epoch
	Eigen::setNbThreads(2);
    // int n = 1;
    // Eigen::setNbThreads(atoi(argv[n++]));
	cout<<"Number of threads: "<<Eigen::nbThreads()<<endl<<endl;	
	
	cout<<"Hyper-parameters: "<<endl<<endl;
	
	// MLP parameters
	int nHLayer = 3; // min = 1, otherwise it is not a MLP
	cout<<"Number of hidden layers: "<<nHLayer<<endl;	
	int maxout_units = 128;
	cout<<"Maxout units: "<<maxout_units<<endl;
	int maxout_pieces = 4;
	cout<<"Maxout pieces: "<<maxout_pieces<<endl;
	
	// SGD parameters
	double learningRate = .1;
	cout<<"Learning rate: "<<learningRate<<endl;
	double LR_decay = .99;
	cout<<"Learning rate decay factor: "<<LR_decay<<endl;
	double LR_min = .01;
	cout<<"Learning rate minimum: "<<LR_min<<endl;
	double momentum = .9;
	cout<<"Momentum: "<<momentum<<endl;
	int momentum_sat = 100;
	cout<<"Momentum saturate: "<<momentum_sat<<endl;
	double momentum_fin = .9;
	cout<<"Momentum final value: "<<momentum_fin<<endl;
	int earlyStop = 100;
	cout<<"Early stop: "<<earlyStop<<endl;
	int batch_size = 100;
	cout<<"Batch size: "<<batch_size<<endl;
	double dropout_input = .8;
	cout<<"Dropout input layer: "<<dropout_input<<endl;
	double dropout_hidden = .5;
	cout<<"Dropout hidden layers: "<<dropout_hidden<<endl;
	double max_col_norm = 2.;
	cout<<"Max column norm: "<<max_col_norm<<endl;
	
	// load trainingSet
	dataSet *trainingSet; //create data set pointer
	trainingSet = new dataSet;
	trainingSet->loadPatterns("/data/lisa/data/mnist/train-images-idx3-ubyte", batch_size);
	trainingSet->loadTargets("/data/lisa/data/mnist/train-labels-idx1-ubyte", batch_size);
	
	// split training set to create validation set
	dataSet *validationSet = trainingSet->split(batch_size);

	// load testSet
	dataSet *testSet; //create data set pointer
	testSet = new dataSet;
	testSet->loadPatterns("/data/lisa/data/mnist/t10k-images-idx3-ubyte", batch_size);
	testSet->loadTargets("/data/lisa/data/mnist/t10k-labels-idx1-ubyte", batch_size);
	
	// neural network creation
	neuralNetwork* NN; // create a neural network pointer
	NN = new neuralNetwork(nHLayer, 784, 10, maxout_units, maxout_pieces, 
		batch_size, dropout_input, dropout_hidden, max_col_norm);
	
	// neural network training
	trainNetwork(NN, trainingSet, validationSet, testSet, 
		learningRate, LR_decay, LR_min, 
		earlyStop, 
		momentum, momentum_sat, momentum_fin);
	
	cout << endl << "-- END OF PROGRAM --" << endl << endl;
	return 1;
}
