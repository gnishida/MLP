#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "HiddenLayer.h"
#include "LogisticRegression.h"

using namespace std;
using namespace cv;

class MLP {
private:
	HiddenLayer* hiddenLayer;
	LogisticRegression* logRegressionLayer;

public:
	MLP(const Mat_<double>& X, const Mat_<double>& Y, int n_in, int n_hidden, int n_out);

	cv::Mat_<double> params();
	void train(const Mat_<double>& X, const Mat_<double>& Y, double lambda, double alpha, int maxIter);
	cv::Mat_<double> predict(const Mat_<double>& input);

	/*
	Updates computeNumericalGradient(double lambda, double beta, double sparsityParam);
	string encodeParams();
	vector<double> encodeDerivatives(const Updates& updates);
	*/

};

