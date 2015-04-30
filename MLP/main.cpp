#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "MLP.h"

using namespace std;

cv::Mat_<double> func(cv::Mat_<double> x) {
	cv::Mat_<double> ret(1, 1);
	ret(0, 0) = sin(x(0, 0) * 4.0);
	return ret;
}

int main() {
	const int N = 100;
	const int NS = 8;
	cv::Mat_<double> X(NS, 1);
	cv::Mat_<double> Y(NS, 1);
	for (int i = 0; i < NS; ++i) {
		double x1 = (double)i / NS * 2 - 1;
		X(i, 0) = x1;
		cv::Mat_<double> x = (cv::Mat_<double>(1, 1) << x1);
		cv::Mat_<double> y = func(x);
		Y(i, 0) = y(0, 0);
	}

	cout << X << endl;
	cout << Y << endl;

	cv::Mat_<uchar> img = cv::Mat_<uchar>::ones(100, 100) * 255;

	MLP mlp(X, 1, 1, 1);
	for (int i = 0; i < N; ++i) {
		cv::Mat_<double> x = (cv::Mat_<double>(1, 1) << (double)i / N * 2 - 1);
		cv::Mat_<double> y = mlp.predict(x);

		cv::Mat_<double> t = func(x);

		cout << x(0, 0) << "," << y(0, 0) << endl;

		
		img((int)((y(0, 0) + 2) * 25), (int)((x(0, 0) + 1) * 50)) = 0;
		img((int)((t(0, 0) + 2) * 25), (int)((x(0, 0) + 1) * 50)) = 128;
	}

	cv::flip(img, img, 0);
	cv::imwrite("test.png", img);

	return 0;
}
