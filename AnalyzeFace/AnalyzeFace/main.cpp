// eigenface.c, by Robin Hewitt, 2007
//
// Example program showing how to implement eigenface with OpenCV

// Usage:
//
// First, you need some face images. I used the ORL face database.
// You can download it for free at
//    www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
//
// List the training and test face images you want to use in the
// input files train.txt and test.txt. (Example input files are provided
// in the download.) To use these input files exactly as provided, unzip
// the ORL face database, and place train.txt, test.txt, and eigenface.exe
// at the root of the unzipped database.
//
// To run the learning phase of eigenface, enter
//    eigenface train
// at the command prompt. To run the recognition phase, enter
//    eigenface test

#include "stdafx.h"
#include "fuction.h"

////定义几个重要的全局变量
IplImage ** faceImgArr        = 0; // 指向训练人脸和测试人脸的指针（在学习和识别阶段指向不同）
CvMat    *  personNumTruthMat = 0; // 人脸图像的ID号
int nTrainFaces               = 0; // 训练图像的数目
int nEigens                   = 0; // 自己取的主要特征值数目
IplImage * pAvgTrainImg       = 0; // 训练人脸数据的平均值
IplImage ** eigenVectArr      = 0; // 投影矩阵，也即主特征向量
CvMat * eigenValMat           = 0; // 特征值
CvMat * projectedTrainFaceMat = 0; // 训练图像的投影


//// 函数原型




//主函数，功能：添加新的脸，批量添加新脸，识别
void main( int argc, char** argv )
{
	// validate that an input was specified
	if( argc < 3 )
	{
		printUsage();
		return;
	}
	//通过判断命令行参数分别执行学习和识别代码
	if( !strcmp(argv[1], "addFace") ) learn();
	else if( !strcmp(argv[1], "test") ) recognize();
	else
	{
		printf("Unknown command: %s\n", argv[1]);
		printUsage();
	}
}


//
void printUsage()
{
	printf("Usage: analyze addFace filepath || analyze addFace addFaces filepath || analyze recognition filepath\n");

}








