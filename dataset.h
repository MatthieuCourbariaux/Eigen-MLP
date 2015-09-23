
#ifndef _DATASET
#define _DATASET

#include <Eigen/Core>
using Eigen::MatrixXd;

//dataSet class
class dataSet
{
	
public:

	int size;
	int nTarget;
	int nPattern;
	MatrixXd** pattern; //input patterns
	MatrixXd** target; //target results

	dataSet(){}

	void loadPatterns(const char* filename, int batch_size);
	void loadTargets(const char* filename, int batch_size);
	int ReverseInt(int i);
	dataSet* split(int batch_size);
};

#endif
