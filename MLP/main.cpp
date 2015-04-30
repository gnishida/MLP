#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "MLP.h"
#include <random>

using namespace std;

cv::Mat_<double> func(cv::Mat_<double> x) {
	cv::Mat_<double> ret(1, 1);
	ret(0, 0) = x(0, 0) * 2 + x(0, 1) * 0.5 - x(0, 2) - x(0,3) * 0.1;
	//ret(0, 0) = x(0, 0) * 0.5 + x(0, 1);
	//ret(0, 0) = sin(x(0, 0) * 4.0);
	return ret;
}

int main() {
	std::mt19937 mt(100);
	std::uniform_real_distribution<double> distribution(-1, 1);

	const int N = 5;
	const int NS = 100;
	cv::Mat_<double> X(NS, 4);
	cv::Mat_<double> Y(NS, 1);
	for (int i = 0; i < NS; ++i) {
		cv::Mat_<double> x(1, X.cols);
		for (int c = 0; c < x.cols; ++c) {
			X(i, c) = distribution(mt);
		}
		cv::Mat_<double> y = func(X.row(i));
		Y(i, 0) = y(0, 0);
	}

	//cout << X << endl;
	//cout << Y << endl;

	cv::Mat_<uchar> img = cv::Mat_<uchar>::ones(100, 100) * 255;

	MLP mlp(X, Y, X.cols, 3, Y.cols);
	mlp.train(X, Y, 0.01, 0.001, 200);

	for (int i = 0; i < N; ++i) {
		cv::Mat_<double> x(1, X.cols);
		for (int c = 0; c < x.cols; ++c) {
			x(0, c) = distribution(mt);
		}
		cv::Mat_<double> y = mlp.predict(x);

		cv::Mat_<double> t = func(x);

		cout << t(0, 0) << ", " << y(0, 0) << endl;

		//img((int)((y(0, 0) + 2) * 25), (int)((x(0, 0) + 1) * 50)) = 0;
		//img((int)((t(0, 0) + 2) * 25), (int)((x(0, 0) + 1) * 50)) = 128;
	}

	cv::flip(img, img, 0);
	cv::imwrite("test.png", img);

	return 0;
}
