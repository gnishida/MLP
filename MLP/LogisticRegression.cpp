#include "LogisticRegression.h"

using namespace std;

LogisticRegression::LogisticRegression(cv::Mat_<double> input, int n_in, int n_out) {
	this->input = input;
	this->n_in = n_in;
	this->n_out = n_out;

	W = cv::Mat_<double>(n_in, n_out);
	cv::randu(W, -sqrt(6.0 / (n_in + n_out)), sqrt(6.0 / (n_in + n_out)));

	b = cv::Mat_<double>::zeros(1, n_out);

	predict(input);
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

/**
 * inputデータからoutputデータを計算し、outputを更新すると共に、それを返却する。
 * inputデータは、K次元の入力データがNサンプルある時、NxKの行列で表現される。
 * つまり、各行が各サンプルに相当する。
 * 一方、outputは、L次元の出力データがNサンプル分ある時、NxLの行列で表現される。
 *
 * @param input		inputデータ
 * @return			outputデータ
 */
cv::Mat_<double> LogisticRegression::predict(const cv::Mat_<double>& input) {
	cv::Mat_<double> e;
	cv::exp(-input * W - cv::repeat(b, input.rows, 1), e);
	output = 1.0 / (1.0 + e);
	return output;
}

double LogisticRegression::negative_log_likelihood(const cv::Mat_<double>& input, const cv::Mat_<double>& target) {
	cv::Mat_<double> log_y_hat, log_one_minus_y_hat;
	cv::log(output, log_y_hat);
	cv::log(1 - output, log_one_minus_y_hat);

	cv::Mat_<double> cross_entropy;
	cv::reduce(-output.mul(log_y_hat) - (1 - output).mul(log_one_minus_y_hat), cross_entropy, 1, CV_REDUCE_SUM);
	cv::reduce(cross_entropy, cross_entropy, 0, CV_REDUCE_AVG);

	return cross_entropy(0, 0);
}

/**
 * outputデータでの誤差に基づいて、W、bを更新する。
 * また、入力データでの誤差を計算して返却する。
 *
 * @param delta		output誤差
 * @param lambda	正規化項の係数
 * @param alpha		学習速度
 * @return			input誤差
 */
void LogisticRegression::grad(const cv::Mat_<double>& delta, double lambda, cv::Mat_<double>& dW, cv::Mat_<double>& db) {
	dW = cv::Mat_<double>::zeros(W.size());
	db = cv::Mat_<double>::zeros(b.size());

	// dW, dbを計算する
	int N = delta.rows;
	for (int r = 0; r < dW.rows; ++r) {
		for (int c = 0; c < dW.cols; ++c) {
			for (int i = 0; i < N; ++i) {
				dW(r, c) -= delta(i, c) * input(i, r);
			}
			dW(r, c) += lambda * W(r, c);
		}
	}
	for (int c = 0; c < dW.cols; ++c) {
		for (int i = 0; i < N; ++i) {
			db(0, c) -= delta(i, c);
		}
		db(0, c) += lambda * b(0, c);
	}
}
