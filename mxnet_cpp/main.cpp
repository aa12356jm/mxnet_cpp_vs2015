#include "imageClassifier.h"


int main()
{
	
	std::string json_file = "./model/mxnet_oneClick/lenetweights-symbol.json";
	std::string param_file = "./model/mxnet_oneClick/lenetweights-0040.params";
	std::string synset_file = "./model/mxnet_oneClick/synset.txt";
	std::string nd_file = "./model/mxnet_oneClick/mean.bin";  //这个均值文件非必须，训练阶段自动生成，测试阶段mean_224.nd可以不存在,但是变量一定要定义   
	
	imageClassifier imgDetect(json_file,param_file, nd_file, synset_file,false);
	cv::Mat img = cv::imread("7_2.jpg");
	std::vector<classifyResult> results = imgDetect.classifier(img);
	
	return 0;
}