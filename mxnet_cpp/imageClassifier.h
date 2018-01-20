#pragma once
#include <stdio.h>

// Path for c_predict_api
#include "mxnet/c_predict_api.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct classifyResult
{
	std::string name;
	float score;
};

class __declspec(dllexport) imageClassifier
{
public:
	imageClassifier(std::string json_file, std::string param_file, std::string nd_file, std::string synset_file, bool useGpu);
	std::vector<classifyResult> classifier(cv::Mat img);

private:
	void GetImageMat(cv::Mat im_ori, mx_float* image_data, const int channels, const cv::Size resize_size, const mx_float* mean_data = nullptr);
	std::vector<std::string> LoadSynset(std::string synset_file);
	PredictorHandle m_pred_hnd;
	NDListHandle m_mean_nd_hnd;
	int m_imgWidth;
	int m_imgHeight;
	int m_imgChannel;
	const mx_float* m_nd_data;
	std::vector<std::string> m_synset;
};

