#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;

namespace myutil {

vector<string> split(const std::string &str, char sep);
void split(const cv::Mat_<double>& data, float ratio1, cv::Mat_<double>& data1, cv::Mat_<double>& data2);
void split(const cv::Mat_<double>& data, float ratio1, float ratio2, cv::Mat_<double>& data1, cv::Mat_<double>& data2, cv::Mat_<double>& data3);
bool loadData(char* filename, cv::Mat_<double>& X, cv::Mat_<double>& Y);
void normalize(cv::Mat_<double> mat, cv::Mat_<double>& normalized_mat, cv::Mat_<double>& mu, cv::Mat_<double>& abs_max);

}