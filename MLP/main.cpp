#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "MLP.h"
#include <vector>
#include <fstream>
#include <iostream>
#include "MyUtil.h"

using namespace std;

cv::Mat_<double> func(cv::Mat_<double> x) {
	cv::Mat_<double> ret(1, 1);
	ret(0, 0) = x(0, 0) * x(0, 1) - x(0, 2) * x(0, 2) + x(0, 3) * 0.1;
	//ret(0, 0) = x(0, 0) * 2 + x(0, 1) * 0.5 - x(0, 2) - x(0,3) * 0.1;
	//ret(0, 0) = x(0, 0) * 0.5 + x(0, 1);
	//ret(0, 0) = sin(x(0, 0) * 4.0);
	return ret;
}

int main(int argc,char *argv[]) {
	if (argc < 2) {
		cout << endl;
		cout << "Usage: " << argv[0] << " <filename>" << endl;
		cout << endl;

		return -1;
	}


	/*
	FILE* fp = fopen("smalldata.txt", "w");

	const int N = 100;
	const int NS = 5; // 2000
	cv::Mat_<double> X(NS, 4);
	cv::Mat_<double> Y(NS, 1);
	for (int i = 0; i < NS; ++i) {
		fprintf(fp, "[");
		cv::Mat_<double> x(1, X.cols);
		for (int c = 0; c < x.cols; ++c) {
			if (c > 0) fprintf(fp, ",");
			X(i, c) = distribution(mt);
			fprintf(fp, "%lf", X(i, c));
		}
		cv::Mat_<double> y = func(X.row(i));
		Y(i, 0) = y(0, 0);

		fprintf(fp, "],[%lf]\n", Y(i, 0));
	}
	fclose(fp);
	*/

	
	cv::Mat_<double> X;
	cv::Mat_<double> Y;
	myutil::loadData(argv[1], X, Y);


	cv::Mat_<double> trainX, trainY, validX, validY, testX, testY;
	myutil::split(X, 0.8f, 0.1f, trainX, validX, testX);
	myutil::split(Y, 0.8f, 0.1f, trainY, validY, testY);



	//cout << X << endl;
	//cout << Y << endl;

	vector<double> lambda; 
	//lambda.push_back(0.001);
	//lambda.push_back(0.0001);
	lambda.push_back(0.00001);

	vector<double> alpha;
	//alpha.push_back(0.001);
	//alpha.push_back(0.0001);
	alpha.push_back(0.00001);

	vector<int> hiddenSize;
	hiddenSize.push_back(3);
	//hiddenSize.push_back(4);
	//hiddenSize.push_back(5);

	vector<int> maxIter;
	//maxIter.push_back(10000);
	maxIter.push_back(50000);
	//maxIter.push_back(100000);
	//maxIter.push_back(500000);

	double best_alpha, best_lambda;
	int best_hiddenSize, best_maxIter;

	double best_score = std::numeric_limits<double>::max();
	for (int i = 0; i < lambda.size(); ++i) {
		for (int j = 0; j < alpha.size(); ++j) {
			for (int k = 0; k < hiddenSize.size(); ++k) {
				for (int l = 0; l < maxIter.size(); ++l) {
					cout << "lambda: " << lambda[i] << ", alpha: " << alpha[j] << ", hiddenSize: " << hiddenSize[k] << ", maxIter: " << maxIter[l] << " ..." << endl;

					MLP mlp(trainX, trainY, hiddenSize[k]);
					mlp.train(trainX, trainY, lambda[i], alpha[j], maxIter[l]);
					double score = mlp.cost(validX, validY, 0);
					cout << "    Score: " << score << endl;

					if (score < best_score) {
						best_score = score;
						best_lambda = lambda[i];
						best_alpha = alpha[j];
						best_hiddenSize = hiddenSize[k];
						best_maxIter = maxIter[l];
					}
				}
			}
		}
	}

	cout << "Best hyperparameters:" << endl;
	cout << "lambda: " << best_lambda << ", alpha: " << best_alpha << ", hiddenSize: " << best_hiddenSize << ", maxIter: " << best_maxIter << endl;
	cout << "    Score: " << best_score << endl;

	MLP mlp(trainX, trainY, best_hiddenSize);
	mlp.train(trainX, trainY, best_lambda, best_alpha, best_maxIter);

	cout << "Performance on the test data:" << endl;
	double error = 0.0;
	for (int i = 0; i < testX.rows; ++i) {
		cv::Mat_<double> y = mlp.predict(testX.row(i));

		error += cv::norm(y - testY.row(i));

		cout << y(0, 0) << ", " << testY(i, 0) << ", " << abs(y(0, 0) - testY(i, 0)) << endl;
	}
	cout << "Avg error: " << error / testX.rows << endl;


	FILE* fp = fopen("result.txt", "w");
	fprintf(fp, "Best hyperparameters for %s:\n", argv[1]);
	fprintf(fp, "lambda: %lf\n", best_lambda);
	fprintf(fp, "alpha: %lf\n", best_alpha);
	fprintf(fp, "hiddenSize: %d\n", best_hiddenSize);
	fprintf(fp, "maxIter: %d\n", best_maxIter);
	fprintf(fp, "Score: %lf\n", best_score);
	fprintf(fp, "Avg error on test data: %lf\n", error / testX.rows);
	fclose(fp);



	return 0;
}
