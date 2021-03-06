﻿#include "MyUtil.h"

namespace myutil {

vector<string> split(const string &str, char sep) {
    vector<string> v;
    stringstream ss(str);
    string buffer;
    while (getline(ss, buffer, sep)) {
        v.push_back(buffer);
    }
    return v;
}

/**
 * データを、指定された比率に従い、2つのデータに分割する。
 *
 */
void split(const cv::Mat_<double>& data, float ratio1, cv::Mat_<double>& data1, cv::Mat_<double>& data2) {
	int rows1 = data.rows * ratio1;
	int rows2 = data.rows - rows1;

	data1 = cv::Mat_<double>(rows1, data.cols);
	data2 = cv::Mat_<double>(rows2, data.cols);

	for (int r = 0; r < data.rows; ++r) {
		for (int c = 0; c < data.cols; ++c) {
			if (r < rows1) {
				data1(r, c) = data(r, c);
			} else {
				data2(r - rows1, c) = data(r, c);
			}
		}
	}
}

void split(const cv::Mat_<double>& data, float ratio1, float ratio2, cv::Mat_<double>& data1, cv::Mat_<double>& data2, cv::Mat_<double>& data3) {
	int rows1 = data.rows * ratio1;
	int rows2 = data.rows * ratio2;
	int rows3 = data.rows - rows1 - rows2;

	data1 = cv::Mat_<double>(rows1, data.cols);
	data2 = cv::Mat_<double>(rows2, data.cols);
	data3 = cv::Mat_<double>(rows3, data.cols);

	for (int r = 0; r < data.rows; ++r) {
		for (int c = 0; c < data.cols; ++c) {
			if (r < rows1) {
				data1(r, c) = data(r, c);
			} else if (r < rows1 + rows2) {
				data2(r - rows1, c) = data(r, c);
			} else {
				data3(r - rows1 - rows2, c) = data(r, c);
			}
		}
	}
}

/**
 * ファイルからデータを読み込む。
 * ファイルの各行は、[x1, x2, ...],[y1, y2, ...] のフォーマットになっていること。
 * [x1, x2, ...]は、行列Xへ格納され、[y1, y2, ...]は、行列Yへ格納される。
 * もしファイルがオープンできない場合は、falseを返却する。
 * 
 */
bool loadData(char* filename, cv::Mat_<double>& X, cv::Mat_<double>& Y) {
	std::ifstream ifs(filename);
    std::string str;
    if (ifs.fail()) {
        return false;
    }


	vector<vector<double> > vecX;
	vector<vector<double> > vecY;

    while (getline(ifs, str)) {
		str = str.substr(1, str.length() - 2);
		vector<string> list = myutil::split(str, ']');

		vector<double> recX, recY;
		vector<string> x_list = myutil::split(list[0], ',');
		for (int i = 0; i < x_list.size(); ++i) {
			recX.push_back(stof(x_list[i]));
		}
		vector<string> y_list = myutil::split(list[1].substr(2), ',');
		for (int i = 0; i < y_list.size(); ++i) {
			recY.push_back(stof(y_list[i]));
		}

		vecX.push_back(recX);
		vecY.push_back(recY);
    }


	X = cv::Mat_<double>(vecX.size(), vecX[0].size());
	Y = cv::Mat_<double>(vecY.size(), vecY[0].size());

	for (int r = 0; r < vecX.size(); ++r) {
		for (int c = 0; c < vecX[r].size(); ++c) {
			X(r, c) = vecX[r][c];
		}
		for (int c = 0; c < vecY[r].size(); ++c) {
			Y(r, c) = vecY[r][c];
		}
	}

	return true;
}

void normalize(cv::Mat_<double> mat, cv::Mat_<double>& normalized_mat, cv::Mat_<double>& mu, cv::Mat_<double>& abs_max) {
	// normalization
	cv::reduce(mat, mu, 0, CV_REDUCE_AVG);
	normalized_mat = mat - cv::repeat(mu, mat.rows, 1);

	// [-1, 1]にする
	cv::reduce(cv::abs(normalized_mat), abs_max, 0, CV_REDUCE_MAX);
	normalized_mat /= cv::repeat(abs_max, mat.rows, 1);
}

}