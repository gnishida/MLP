#include "HiddenLayer.h"
#include <math.h>

HiddenLayer::HiddenLayer(cv::Mat_<double> input, int n_in, int n_out) {
	W = cv::Mat_<double>(n_in, n_out);
	cv::randu(W, -sqrt(6.0 / (n_in + n_out)), sqrt(6.0 / (n_in + n_out)));

	b = cv::Mat_<double>::zeros(1, n_out);

	init(input, n_in, n_out, W, b);
}

HiddenLayer::HiddenLayer(cv::Mat_<double> input, int n_in, int n_out, cv::Mat_<double> W, cv::Mat_<double> b) {
	init(input, n_in, n_out, W, b);
}

/**
 * パラメータW, bを１つの行ベクトルにして返却する。
 *
 * @return		パラメータ群（行ベクトル）
 */
cv::Mat_<double> HiddenLayer::params() {
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

cv::Mat_<double> HiddenLayer::predict(const cv::Mat_<double>& input) {
	return mat_tanh(input * W + cv::repeat(b, input.rows, 1));
}

void HiddenLayer::init(cv::Mat_<double> input, int n_in, int n_out, cv::Mat_<double> W, cv::Mat_<double> b) {
	this->input = input;
	this->n_in = n_in;
	this->n_out = n_out;
	this->W = W;
	this->b = b;

	output = mat_tanh(input * W + cv::repeat(b, input.rows, 1));
}

cv::Mat_<double> HiddenLayer::mat_tanh(const cv::Mat_<double>& mat) {
	cv::Mat_<double> ret(mat.size());

	for (int r = 0; r < mat.rows; ++r) {
		for (int c = 0; c < mat.cols; ++c) {
			ret(r, c) = tanh(mat(r, c));
		}
	}
	return ret;
}