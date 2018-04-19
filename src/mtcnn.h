//author: samylee
//date: 2018.4.19

#ifndef MTCNN_H
#define MTCNN_H
#include "network.h"

class Pnet
{
public:
	Pnet();
	~Pnet();
	void run(Mat &image, float scale);
	void freePnetSpace();

	float nms_threshold;
	mydataFmt Pthreshold;
	bool firstFlag;
	vector<struct Bbox> boundingBox_;
	vector<orderScore> bboxScore_;
private:
	//the image for mxnet conv
	struct pBox *rgb;
	struct pBox *conv1_matrix;
	//the 1th layer's out conv
	struct pBox *conv1;
	struct pBox *maxPooling1;
	struct pBox *maxPooling_matrix;
	//the 3th layer's out
	struct pBox *conv2;
	struct pBox *conv3_matrix;
	//the 4th layer's out   out
	struct pBox *conv3;
	struct pBox *score_matrix;
	//the 4th layer's out   out
	struct pBox *score_;
	//the 4th layer's out   out
	struct pBox *location_matrix;
	struct pBox *location_;

	//Weight
	struct Weight *conv1_wb;
	struct pRelu *prelu_gmma1;
	struct Weight *conv2_wb;
	struct pRelu *prelu_gmma2;
	struct Weight *conv3_wb;
	struct pRelu *prelu_gmma3;
	struct Weight *conv4c1_wb;
	struct Weight *conv4c2_wb;

	void generateBbox(const struct pBox *score, const struct pBox *location, mydataFmt scale);
};

class Rnet
{
public:
	Rnet();
	~Rnet();
	float Rthreshold;
	void run(Mat &image);
	struct pBox *score_;
	struct pBox *location_;
private:
	struct pBox *rgb;

	struct pBox *conv1_matrix;
	struct pBox *conv1_out;
	struct pBox *pooling1_out;

	struct pBox *conv2_matrix;
	struct pBox *conv2_out;
	struct pBox *pooling2_out;

	struct pBox *conv3_matrix;
	struct pBox *conv3_out;

	struct pBox *fc4_out;

	//Weight
	struct Weight *conv1_wb;
	struct pRelu *prelu_gmma1;
	struct Weight *conv2_wb;
	struct pRelu *prelu_gmma2;
	struct Weight *conv3_wb;
	struct pRelu *prelu_gmma3;
	struct Weight *fc4_wb;
	struct pRelu *prelu_gmma4;
	struct Weight *score_wb;
	struct Weight *location_wb;

	void RnetImage2MatrixInit(struct pBox *pbox);
};

class Onet
{
public:
	Onet();
	~Onet();
	void run(Mat &image);
	float Othreshold;
	struct pBox *score_;
	struct pBox *location_;
	struct pBox *keyPoint_;
private:
	struct pBox *rgb;
	struct pBox *conv1_matrix;
	struct pBox *conv1_out;
	struct pBox *pooling1_out;

	struct pBox *conv2_matrix;
	struct pBox *conv2_out;
	struct pBox *pooling2_out;

	struct pBox *conv3_matrix;
	struct pBox *conv3_out;
	struct pBox *pooling3_out;

	struct pBox *conv4_matrix;
	struct pBox *conv4_out;

	struct pBox *fc5_out;

	//Weight
	struct Weight *conv1_wb;
	struct pRelu *prelu_gmma1;
	struct Weight *conv2_wb;
	struct pRelu *prelu_gmma2;
	struct Weight *conv3_wb;
	struct pRelu *prelu_gmma3;
	struct Weight *conv4_wb;
	struct pRelu *prelu_gmma4;
	struct Weight *fc5_wb;
	struct pRelu *prelu_gmma5;
	struct Weight *score_wb;
	struct Weight *location_wb;
	struct Weight *keyPoint_wb;
	void OnetImage2MatrixInit(struct pBox *pbox);
};

class mtcnn
{
public:
	mtcnn();
	void findFace(Mat &image, int min_size);
private:
	Mat reImage;
	float nms_threshold[3];
	int MIN_DET_SIZE;
	float factor;
	
	Pnet proposeNet;
	vector<struct Bbox> firstBbox_;
	vector<struct orderScore> firstOrderScore_;
	Rnet refineNet;
	vector<struct Bbox> secondBbox_;
	vector<struct orderScore> secondBboxScore_;
	Onet outNet;
	vector<struct Bbox> thirdBbox_;
	vector<struct orderScore> thirdBboxScore_;
};

#endif
