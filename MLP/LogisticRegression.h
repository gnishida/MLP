#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

class LogisticRegression {
public:
	int n_in;
	int n_out;
	cv::Mat_<double> input;
	cv::Mat_<double> output;
	cv::Mat_<double> W;
	cv::Mat_<double> b;

public:
	LogisticRegression(cv::Mat_<double> input, int n_in, int n_out);

	cv::Mat_<double> params();
	cv::Mat_<double> predict(const cv::Mat_<double>& input);
	double negative_log_likelihood(const cv::Mat_<double>& input, const cv::Mat_<double>& target);
	void grad(const cv::Mat_<double>& delta, double lambda, cv::Mat_<double>& dW, cv::Mat_<double>& db);
	cv::Mat_<double> back_propagation(const cv::Mat_<double>& delta, double lambda, double alpha);

private:
	
};

