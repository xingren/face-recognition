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

////���弸����Ҫ��ȫ�ֱ���
IplImage ** faceImgArr        = 0; // ָ��ѵ�������Ͳ���������ָ�루��ѧϰ��ʶ��׶�ָ��ͬ��
CvMat    *  personNumTruthMat = 0; // ����ͼ���ID��
int nTrainFaces               = 0; // ѵ��ͼ�����Ŀ
int nEigens                   = 0; // �Լ�ȡ����Ҫ����ֵ��Ŀ
IplImage * pAvgTrainImg       = 0; // ѵ���������ݵ�ƽ��ֵ
IplImage ** eigenVectArr      = 0; // ͶӰ����Ҳ������������
CvMat * eigenValMat           = 0; // ����ֵ
CvMat * projectedTrainFaceMat = 0; // ѵ��ͼ���ͶӰ


//// ����ԭ��




//�����������ܣ�����µ������������������ʶ��
void main( int argc, char** argv )
{
	// validate that an input was specified
	if( argc < 3 )
	{
		printUsage();
		return;
	}
	//ͨ���ж������в����ֱ�ִ��ѧϰ��ʶ�����
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








