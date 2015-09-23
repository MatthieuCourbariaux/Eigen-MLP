
#ifndef NNLayer
#define NNLayer

#include <Eigen/Core>
using Eigen::MatrixXd;
using namespace std;

class layer
{

public:
	
	int n_inputs;
	int n_units;
	
	int batch_size;
	double activation_rate;
	double scale;
	double max_col_norm;
	
	MatrixXd w; // weight matrix
	MatrixXd w_best;
	MatrixXd dEdw; 
	MatrixXd update_w;
	
	MatrixXd b; // bias vector
	MatrixXd b_best;
	MatrixXd dEdb; 
	MatrixXd update_b;
	
	MatrixXd* x;
	MatrixXd y;
	
	MatrixXd* dEdx;
	
	MatrixXd z;
	MatrixXd dEdz;
	
	MatrixXd dropout_mask;
	
	layer(){}
	~layer(){}
	
	void save();
	void load();
	void fprop_weighted_sum(bool test);
	void bprop_weighted_sum();
	void update(double LR, double momentum);
	
	// dropout
	void dropout();

	// L2 constraint on the inputs of each unit
	void column_normalization();
};

class maxout_layer: public layer
{

public:
	
	int n_pieces;
	MatrixXd dEdy;
	
	maxout_layer(){}
	~maxout_layer(){}
	
	maxout_layer(int p_n_inputs, int p_n_units, int p_n_pieces,
			MatrixXd* x, MatrixXd* dEdx,
			int p_batch_size, double p_activation_rate, double p_scale, double p_max_col_norm);
	
	void maxout();
	void maxout_derivative();
	void init();
	void fprop(bool test);
	void bprop(double LR, double momentum);
};

class softmax_layer: public layer
{

public:
	
	softmax_layer(){}
	~softmax_layer(){}
	
	softmax_layer(int p_n_inputs, int p_n_units,
			MatrixXd* x, MatrixXd* dEdx,
			int p_batch_size, double p_activation_rate, double p_scale, double p_max_col_norm);
	
	void softmax();
	void init();
	void fprop(bool test);
	void bprop(MatrixXd* t, double LR, double momentum);
	
	// cost functions
	int getLineMaxIndex(MatrixXd* X, int line);
	double nll_sum(MatrixXd* t);
};

#endif
