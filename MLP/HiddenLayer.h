#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

class HiddenLayer {
public:
	int n_in;
	int n_out;
	cv::Mat_<double> input;
	cv::Mat_<double> output;
	cv::Mat_<double> W;
	cv::Mat_<double> b;

public:
	HiddenLayer();

	void init(int n_in, int n_out);
	cv::Mat_<double> predict(const cv::Mat_<double>& input);
	void grad(const cv::Mat_<double>& delta, double lambda, cv::Mat_<double>& dW, cv::Mat_<double>& db);

private:
	cv::Mat_<double> mat_tanh(const cv::Mat_<double>& mat);
};

