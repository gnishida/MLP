#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "HiddenLayer.h"
#include "LogisticRegression.h"

using namespace std;
using namespace cv;

struct Updates {
	double cost;
	Mat_<double> dW1;
	Mat_<double> dW2;
	Mat_<double> db1;
	Mat_<double> db2;
};

class MLP {
private:
	HiddenLayer hiddenLayer;
	LogisticRegression logRegressionLayer;

	Mat_<double> W1, W2;		// 重み
	Mat_<double> b1, b2;		// バイアス
	int N;						// 観測データの数
	int hiddenSize;				// hiddenレイヤのユニット数

public:
	MLP(const Mat_<double>& input, int n_in, int n_hidden, int n_out);

	cv::Mat_<double> params();
	Updates train(const Mat_<double>& X, const Mat_<double>& Y, double lambda, double alpha, int maxIter, double sparsityParam);
	cv::Mat_<double> predict(const Mat_<double>& input);



	void decodeAndUpdate(const vector<double>& theta);
	void update(const Updates& updates, double eta);
	void visualize(char* filename);
	Updates computeNumericalGradient(double lambda, double beta, double sparsityParam);
	string encodeParams();
	vector<double> encodeDerivatives(const Updates& updates);

private:
	Updates sparseEncoderCost(const Mat_<double>& W1, const Mat_<double>& W2, const Mat_<double>& b1, const Mat_<double>& b2, double lambda, double beta, double sparsityParam);
	Mat_<double> sigmoid(const Mat_<double>& z);
	void sigmoid(const Mat_<double>& z, Mat_<double>& ret);
	double mat_sum(const Mat_<double>& m);
	double mat_avg(const Mat_<double>& m);
	double mat_max(const Mat_<double>& m);
	double mat_min(const Mat_<double>& m);
};

