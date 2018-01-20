#include "stdafx.h"
#include "imageClassifier.h"


class BufferFile {
public:
	std::string file_path_;
	int length_;
	char* buffer_;

	explicit BufferFile(std::string file_path)
		:file_path_(file_path) {

		std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
		if (!ifs) {
			std::cerr << "Can't open the file. Please check " << file_path << ". \n";
			length_ = 0;
			buffer_ = NULL;
			return;
		}

		ifs.seekg(0, std::ios::end);
		length_ = ifs.tellg();
		ifs.seekg(0, std::ios::beg);
		std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

		buffer_ = new char[sizeof(char) * length_];
		ifs.read(buffer_, length_);
		ifs.close();
	}

	int GetLength() {
		return length_;
	}
	char* GetBuffer() {
		return buffer_;
	}

	~BufferFile() {
		if (buffer_) {
			delete[] buffer_;
			buffer_ = NULL;
		}
	}
};


imageClassifier::imageClassifier(std::string json_file, std::string param_file, std::string nd_file, std::string synset_file,bool useGpu)
{
	BufferFile json_data(json_file);//将json文件和params文件读进来
	BufferFile param_data(param_file);
	//选择是否使用gpu
	int dev_type;
	if (useGpu)
		dev_type = 2;
	else
		dev_type = 1;

	int dev_id = 0;  // arbitrary.
	mx_uint num_input_nodes = 1;  // 1 for feedforward
	const char* input_key[1] = { "data" };
	const char** input_keys = input_key;

	// Image size and channels 这个是设置输入图像的大小，需要根据模型训练时候的大小来设置
	m_imgWidth = 299;
	m_imgHeight = 299;
	m_imgChannel = 3;

	//创建检测器对象
	m_pred_hnd = 0;

	const mx_uint input_shape_indptr[2] = { 0, 4 };
	const mx_uint input_shape_data[4] = { 1,static_cast<mx_uint>(m_imgChannel),
		static_cast<mx_uint>(m_imgHeight),static_cast<mx_uint>(m_imgWidth) };
	
	assert(json_data.GetLength() == 0);
	assert(param_data.GetLength() == 0);

	// Create Predictor
	MXPredCreate((const char*)json_data.GetBuffer(),(const char*)param_data.GetBuffer(),
		static_cast<size_t>(param_data.GetLength()),dev_type,dev_id,num_input_nodes,input_keys,
		input_shape_indptr,input_shape_data,&m_pred_hnd);
	assert(m_pred_hnd);


	//读取均值文件
	m_nd_data = NULL;
	m_mean_nd_hnd = 0;
	BufferFile nd_buf(nd_file);

	if (nd_buf.GetLength() > 0) {
		mx_uint nd_index = 0;
		mx_uint nd_len;
		const mx_uint* nd_shape = 0;
		const char* nd_key = 0;
		mx_uint nd_ndim = 0;

		MXNDListCreate((const char*)nd_buf.GetBuffer(), nd_buf.GetLength(), &m_mean_nd_hnd, &nd_len);
		MXNDListGet(m_mean_nd_hnd, nd_index, &nd_key, &m_nd_data, &nd_shape, &nd_ndim);
	}
	// 存放类别的文件
	m_synset = LoadSynset(synset_file);
}

std::vector<classifyResult> imageClassifier::classifier(cv::Mat img)
{
	int image_size = m_imgWidth * m_imgHeight * m_imgChannel;

	// Read Image Data
	std::vector<mx_float> image_data = std::vector<mx_float>(image_size);

	GetImageMat(img, image_data.data(), m_imgChannel, cv::Size(m_imgWidth, m_imgHeight), m_nd_data);

	// Set Input Image
	MXPredSetInput(m_pred_hnd, "data", image_data.data(), image_size);

	double start = (double)cvGetTickCount();
	// Do Predict Forward
	MXPredForward(m_pred_hnd);

	double time = (double)cvGetTickCount() - start;

	//printf("run time = %gms\n", time / (cvGetTickFrequency() * 1000));//毫秒

	mx_uint output_index = 0;
	mx_uint *shape = 0;
	mx_uint shape_len;

	// Get Output Result
	MXPredGetOutputShape(m_pred_hnd, output_index, &shape, &shape_len);

	size_t size = 1;
	for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];

	std::vector<float> data(size);

	MXPredGetOutput(m_pred_hnd, output_index, &(data[0]), size);

	//输出有几个类别，标签文件中也要有几个类别
	if (data.size() != m_synset.size()) {
		std::cerr << "Result data and synset size does not match!" << std::endl;
	}
	std::vector<classifyResult> results;
	for (int i = 0;i < data.size();i++)
	{
		classifyResult tempResult;
		tempResult.name = m_synset[i];
		tempResult.score = data[i];
		results.push_back(tempResult);
	}
	return results;
}

void imageClassifier::GetImageMat(cv::Mat im_ori, mx_float* image_data, const int channels,const cv::Size resize_size, const mx_float* mean_data)
{
	// Read all kinds of file into a BGR color 3 channels image
	assert(im_ori.empty());

	cv::Mat im;
	resize(im_ori, im, resize_size);
	int size = im.rows * im.cols * channels;

	mx_float* ptr_image_r = image_data;
	mx_float* ptr_image_g = image_data + size / 3;
	mx_float* ptr_image_b = image_data + size / 3 * 2;
	float mean_b, mean_g, mean_r;
	mean_b = 195;
	mean_g = 196;
	mean_r = 196;
	//mean_b = mean_g = mean_r = DEFAULT_MEAN;
	for (int i = 0; i < im.rows; i++) {
		uchar* data = im.ptr<uchar>(i);

		for (int j = 0; j < im.cols; j++) {
			if (mean_data) {
				mean_r = *mean_data;
				if (channels > 1) {
					mean_g = *(mean_data + size / 3);
					mean_b = *(mean_data + size / 3 * 2);
				}
				mean_data++;
			}
			if (channels > 1) {
				*ptr_image_b++ = static_cast<mx_float>(*data++) - mean_b;
				*ptr_image_g++ = static_cast<mx_float>(*data++) - mean_g;
			}
			*ptr_image_r++ = static_cast<mx_float>(*data++) - mean_r;;
		}
	}
}

//加载数据分类的label
std::vector<std::string> imageClassifier::LoadSynset(std::string synset_file) {
	std::ifstream fi(synset_file.c_str());

	if (!fi.is_open()) {
		std::cerr << "Error opening synset file " << synset_file << std::endl;
		assert(false);
	}

	std::vector<std::string> output;

	std::string synset, lemma;
	while (fi >> synset) {
		getline(fi, lemma);
		output.push_back(lemma);
	}
	fi.close();
	return output;
}
