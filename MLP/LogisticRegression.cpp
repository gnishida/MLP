#include "LogisticRegression.h"

LogisticRegression::LogisticRegression(cv::Mat_<double> input, int n_in, int n_out) {
	this->input = input;
	this->n_in = n_in;
	this->n_out = n_out;
}

/**
 * パラメータW, bを１つの行ベクトルにして返却する。
 *
 * @return		パラメータ群（行ベクトル）
 */
cv::Mat_<double> LogisticRegression::params() {
	cv::Mat_<double> ret(1, W.rows * W.cols + b.rows * b.cols);

	int index = 0;
	for (int r = 0; r < W.rows; ++r) {
		for (int c = 0; c < W.cols; ++c) {
			ret(0, index++) = W(r, c);
		}
	}
	for (int r = 0; r < b.rows; ++r) {
		for (int c = 0; c < b.cols; ++c) {
			ret(0, index++) = b(r, c);
		}
	}

	return ret;
}

cv::Mat_<double> LogisticRegression::predict(const cv::Mat_<double>& input) {
	cv::Mat_<double> e;
	cv::exp(-input * W - cv::repeat(b, input.rows, 1), e);
	return 1.0 / (1.0 + e);
}

double LogisticRegression::negative_log_likelihood(const cv::Mat_<double>& input, const cv::Mat_<double>& output) {
	cv::Mat_<double> y_hat = predict(input);

	cv::Mat_<double> log_y_hat, log_one_minus_y_hat;
	cv::log(y_hat, log_y_hat);
	cv::log(1 - y_hat, log_one_minus_y_hat);

	cv::Mat_<double> cross_entropy;
	cv::reduce(-output.mul(log_y_hat) - (1 - output).mul(log_one_minus_y_hat), cross_entropy, 1, CV_REDUCE_SUM);
	cv::reduce(cross_entropy, cross_entropy, 0, CV_REDUCE_AVG);

	return cross_entropy(0, 0);
}

