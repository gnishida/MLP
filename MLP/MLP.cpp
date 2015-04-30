#include "MLP.h"
#include <iostream>
#include <sstream>

/**
 * Neural networkを構築する。
 *
 * @param X				入力データ（各行が、各入力x_iに相当）
 * @param Y				出力データ（各行が、各出力y_iに相当）
 * @param hiddenSize	hiddenレイヤのユニット数
 */
MLP::MLP(const Mat_<double>& X, const Mat_<double>& Y, int n_in, int n_hidden, int n_out) {
	hiddenLayer = new HiddenLayer(X, n_in, n_out);
	logRegressionLayer = new LogisticRegression(hiddenLayer->output, n_hidden, n_out);
}

/**
 * パラメータW, bを１つの行ベクトルにして返却する。
 *
 * @return		パラメータ群（行ベクトル）
 */
cv::Mat_<double> MLP::params() {
	cv::Mat_<double> p1 = hiddenLayer->params();
	cv::Mat_<double> p2 = logRegressionLayer->params();
	cv::Mat_<double> ret(1, p1.cols + p2.cols);
	for (int c = 0; c < p1.cols; ++c) {
		ret(0, c) = p1(0, c);
	}
	for (int c = 0; c < p2.cols; ++c) {
		ret(0, p1.cols + c) = p2(0, c);
	}
	return ret;
}

/**
 * 学習する。
 *
 * @return			コスト
 */
void MLP::train(const Mat_<double>& X, const Mat_<double>& Y, double lambda, double alpha, int maxIter) {
	for (int iter = 0; iter < maxIter; ++iter) {
		cout << "Score: " << logRegressionLayer->negative_log_likelihood(X, Y) << endl;

		cv::Mat_<double> h2 = predict(X);

		cv::Mat_<double> delta = logRegressionLayer->back_propagation(Y - h2, lambda, alpha);
		hiddenLayer->back_propagation(delta, lambda, alpha);
	}
}

cv::Mat_<double> MLP::predict(const Mat_<double>& input) {
	cv::Mat_<double> h = hiddenLayer->predict(input);
	return logRegressionLayer->predict(h);
}

/**
 * 数値計算により関数fの、xにおける勾配を計算し、返却する。
 *
 * @param func		関数fのポインタ
 * @param x			このポイントにおける勾配を計算する（xは、行ベクトルであること！）
 * @return			勾配ベクトル
 */
/*
Updates MLP::computeNumericalGradient(double lambda, double beta, double sparsityParam) {
	Updates updates;
	updates.dW1 = Mat_<double>(W1.size());
	updates.dW2 = Mat_<double>(W2.size());
	updates.db1 = Mat_<double>(b1.size());
	updates.db2 = Mat_<double>(b2.size());

	double e = 0.0001;

	for (int r = 0; r < W1.rows; ++r) {
		for (int c = 0; c < W1.cols; ++c) {
			Mat_<double> dW1 = Mat_<double>::zeros(W1.size());
			dW1(r, c) = e;

			Updates u1 = sparseEncoderCost(W1 + dW1, W2, b1, b2, lambda, beta, sparsityParam);
			Updates u2 = sparseEncoderCost(W1 - dW1, W2, b1, b2, lambda, beta, sparsityParam);
			updates.dW1(r, c) = (u1.cost - u2.cost) / e / 2.0;
		}
	}

	for (int r = 0; r < W2.rows; ++r) {
		for (int c = 0; c < W2.cols; ++c) {
			Mat_<double> dW2 = Mat_<double>::zeros(W2.size());
			dW2(r, c) = e;

			Updates u1 = sparseEncoderCost(W1, W2 + dW2, b1, b2, lambda, beta, sparsityParam);
			Updates u2 = sparseEncoderCost(W1, W2 - dW2, b1, b2, lambda, beta, sparsityParam);
			updates.dW2(r, c) = (u1.cost - u2.cost) / e / 2.0;
		}
	}

	for (int r = 0; r < b1.rows; ++r) {
		Mat_<double> db1 = Mat_<double>::zeros(b1.size());
		db1(r, 0) = e;

		Updates u1 = sparseEncoderCost(W1, W2, b1 + db1, b2, lambda, beta, sparsityParam);
		Updates u2 = sparseEncoderCost(W1, W2, b1 - db1, b2, lambda, beta, sparsityParam);
		updates.db1(r, 0) = (u1.cost - u2.cost) / e / 2.0;
	}

	for (int r = 0; r < b2.rows; ++r) {
		Mat_<double> db2 = Mat_<double>::zeros(b2.size());
		db2(r, 0) = e;

		Updates u1 = sparseEncoderCost(W1, W2, b1, b2 + db2, lambda, beta, sparsityParam);
		Updates u2 = sparseEncoderCost(W1, W2, b1, b2 - db2, lambda, beta, sparsityParam);
		updates.db2(r, 0) = (u1.cost - u2.cost) / e / 2.0;
	}

	return updates;
}

string MLP::encodeParams() {
	ostringstream oss;

	oss << "[";
	for (int c = 0; c < W1.cols; ++c) {
		for (int r = 0; r < W1.rows; ++r) {
			oss << W1(r, c) << ",";
		}
	}
	for (int c = 0; c < W2.cols; ++c) {
		for (int r = 0; r < W2.rows; ++r) {
			oss << W2(r, c) << ",";
		}
	}
	for (int r = 0; r < b1.rows; ++r) {
		oss << b1(r, 0) << ",";
	}
	for (int r = 0; r < b2.rows; ++r) {
		oss << b2(r, 0);
		if (r < b2.rows - 1) {
			oss << ",";
		}
	}
	oss << "]";

	return oss.str();
}

vector<double> MLP::encodeDerivatives(const Updates& updates) {
	vector<double> ret(W1.rows * W1.cols + W2.rows * W2.cols + b1.rows + b2.rows);
	int index = 0;

	for (int c = 0; c < W1.cols; ++c) {
		for (int r = 0; r < W1.rows; ++r) {
			ret[index++] = updates.dW1(r, c);
		}
	}
	for (int c = 0; c < W2.cols; ++c) {
		for (int r = 0; r < W2.rows; ++r) {
			ret[index++] = updates.dW2(r, c);
		}
	}
	for (int r = 0; r < b1.rows; ++r) {
		ret[index++] = updates.db1(r, 0);
	}
	for (int r = 0; r < b2.rows; ++r) {
		ret[index++] = updates.db2(r, 0);
	}

	return ret;
}


*/