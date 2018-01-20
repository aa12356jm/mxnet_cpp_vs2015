#include "imageClassifier.h"


int main()
{
	
	std::string json_file = "./model/mxnet_oneClick/lenetweights-symbol.json";
	std::string param_file = "./model/mxnet_oneClick/lenetweights-0040.params";
	std::string synset_file = "./model/mxnet_oneClick/synset.txt";
	std::string nd_file = "./model/mxnet_oneClick/mean.bin";  //�����ֵ�ļ��Ǳ��룬ѵ���׶��Զ����ɣ����Խ׶�mean_224.nd���Բ�����,���Ǳ���һ��Ҫ����   
	
	imageClassifier imgDetect(json_file,param_file, nd_file, synset_file,false);
	cv::Mat img = cv::imread("7_2.jpg");
	std::vector<classifyResult> results = imgDetect.classifier(img);
	
	return 0;
}