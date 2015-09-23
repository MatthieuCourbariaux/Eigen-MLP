
#include "train.h"
#include "dataset.h"
#include <iostream>
using namespace std;

// Train the NN using gradient descent
void trainNetwork(neuralNetwork* NN, dataSet* trainingSet, dataSet* validationSet, dataSet* testSet, 
	double learningRate, double LR_decay, double LR_min, 
	int earlyStop, 
	double momentum_start, int momentum_sat, double momentum_fin)
{		
	int stop = 0;
	int epoch = 1;
	int best_epoch;
	double momentum = momentum_start;
	double best_momentum;
	double LR = learningRate;
	double best_LR;
	double validation_error;
	double best_validation_error = 100.;
	double validation_nll;
	double best_validation_nll = 1.;
	double test_error;
	double test_nll;
	double train_error;
	double train_nll;
	
	// initialize the neural network
	NN->init();
	
	// phase 1:
	// train ont the train set until the validation error stop decreasing
	cout << endl << "Training on the training set until the validation error stops decreasing for "<<earlyStop<<" epochs:" << endl;
	
	while (stop < earlyStop)
	{		
		// train the model on the train set
		NN->train(trainingSet, LR, momentum);
		
		// validation test
		NN->test(validationSet,&validation_error, &validation_nll);
		
		// monitoring
		cout <<endl<<"Epoch "<<epoch;
		cout <<endl<<"	learning rate = " << LR;
		cout <<endl<<"	momentum = " << momentum;
		cout <<endl<<"	validation error rate = "<< validation_error <<"%";
		
		// Early stop
		if (validation_error >= best_validation_error) stop += 1;	
		else
		{	
			// continue training for a little while
			stop = 0;
			
			// save the parameters
			best_validation_error = validation_error;
			best_validation_nll = validation_nll;
			best_epoch = epoch;
			best_LR = LR;
			best_momentum = momentum;
			NN->save();
		}
		
		// update momentum
		if(momentum < momentum_fin) momentum = momentum_start + (momentum_fin-momentum_start)/momentum_sat * epoch;
		else momentum = momentum_fin;
		
		// update LR
		if(LR>LR_min) LR *= LR_decay;
		else LR = LR_min;
		
		// update the number of epochs
		epoch = epoch + 1;
	}
	
	// load the best parameters
	NN->init();
	NN->load();
	epoch = best_epoch + 1;
	LR = best_LR;
	momentum = best_momentum;
	validation_nll = best_validation_nll;
	validation_error = best_validation_error;
	
	// get the train NLL
	NN->test(trainingSet,&train_error, &train_nll);
	
	// monitor the results of the first phase
	cout <<endl<<endl<<"Early stop at epoch "<<best_epoch;
	cout <<", validation error rate = "<< validation_error <<"%";
	
	// phase 2:
	// train on both the train and validation set until 
	// the validation nll matches the train nll of phase 1
	cout <<endl<<"Training on both the training and validation sets until the validation NLL, "<<validation_nll<<",  reaches the training NLL, "<<train_nll<<":" << endl;
	
	while(validation_nll>=train_nll)
	{
		// train the model on the train set
		NN->train(trainingSet, LR, momentum);
		
		// train the model on the validation set as well
		NN->train(validationSet, LR, momentum);
		
		// validation test
		NN->test(validationSet,&validation_error, &validation_nll);
		
		// test the model on the test set
		NN->test(testSet,&test_error, &test_nll);
		
		// monitoring
		cout <<endl<<"Epoch "<<epoch;
		cout <<endl<<"	learning rate = " << LR;
		cout <<endl<<"	momentum = " << momentum;
		cout <<endl<<"	validation NLL = "<< validation_nll;
		cout <<endl<<"	test error rate = "<< test_error <<"%";		
		
		// update momentum
		if(momentum < momentum_fin) momentum = momentum_start + (momentum_fin-momentum_start)/momentum_sat * epoch;
		else momentum = momentum_fin;
		
		// update LR
		if(LR>LR_min) LR *= LR_decay;
		else LR = LR_min;
		
		// update the number of epochs
		epoch = epoch + 1;
	}
	
	cout <<endl<<endl<<"Training done"<<endl;
}


