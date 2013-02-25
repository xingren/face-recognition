/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <iostream>
#include <fstream>
#include <sstream>

#define FACE_WIDTH 92
#define FACE_HIGH 112
#define TrainDataPath "../LBP_TrainData.xml"

CvRect** displaydetection(IplImage *pInpImg,int &resCount)
{

	CvHaarClassifierCascade* pCascade=0;		//ָ�������ļ��л�ȡ�ķ�����
	CvMemStorage* pStorage=0;					//�洢��⵽����������
	CvSeq* pFaceRectSeq;
	pStorage=cvCreateMemStorage(0);				//����Ĭ�ϴ���64k�Ķ�̬�ڴ�����

	//��Ҫ����ʵ������޸�:���ط�������·��
	pCascade=(CvHaarClassifierCascade*)cvLoad("../haarcascade_frontalface_alt2.xml");		//���ط�����
	if (!pInpImg||!pStorage||!pCascade)
	{
		printf("initialization failed:%s\n",(!pInpImg)?"can't load image file":(!pCascade)?"can't load haar-cascade---make sure path is correct":"unable to allocate memory for data storage","pass a wrong IplImage*");
		return NULL;
	}
	//�������
	pFaceRectSeq=cvHaarDetectObjects(pInpImg,pCascade,pStorage,
		1.2,2,CV_HAAR_DO_CANNY_PRUNING,cvSize(40,40));
	//����⵽�������Ծ��ο�����
	int i;
	cvNamedWindow("haar window",1);
	printf("the number of face is %d\n",pFaceRectSeq->total);
	if(pFaceRectSeq->total == 0)
		return NULL;

	IplImage *result = cvCreateImage(cvSize(92,112),pInpImg->depth,pInpImg->nChannels);
	resCount = pFaceRectSeq->total;
	//if(NULL == result)
	//	//��Ӵ������޷������ڴ�
	//{}

	IplImage *show_img = cvCreateImage(cvSize(92,112),pInpImg->depth,pInpImg->nChannels);

	char str[100];
	CvRect** res = new CvRect*[resCount];
	for (i=0; i<(pFaceRectSeq?pFaceRectSeq->total:0); i++)
	{

		CvRect* r=(CvRect*)cvGetSeqElem(pFaceRectSeq,i);
		res[i] = new CvRect;
		memcpy(res[i],r,sizeof(CvRect));
		CvPoint pt1= {r->x,r->y};
		CvPoint pt2= {r->x+r->width,r->y+r->height};

		cvSetImageROI(pInpImg,*r);

		//����׽��������

		

		//cvSetImageROI(pInpImg,*r);
		cvResize(pInpImg,result,CV_INTER_LINEAR);
		sprintf(str,"%d",i);
		strcat(str,".jpg");
		cvSaveImage(str,result);

		cvResetImageROI(pInpImg);
		//cvResetImageROI(pInpImg);
		show_img = cvCloneImage(pInpImg);
		cvRectangle(show_img,pt1,pt2,CV_RGB(0,255,0),3,4,0);

	}

	cvShowImage("haar window",show_img);
	//	cvResetImageROI(pInpImg);
//	cvWaitKey(0);
	cvDestroyWindow("haar window");
	//cvReleaseImage(&pInpImg);
	cvReleaseHaarClassifierCascade(&pCascade);
	cvReleaseMemStorage(&pStorage);
	cvReleaseImage(&result);
	cvReleaseImage(&show_img);
	return res;
}
using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, const char *argv[]) {

	//freopen("../train.txt","r",stdin);
    
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
    //0model->train(images, labels);
	model->load("../FaceData.xml");
    // The following line predicts the label of a given
    // test image:
	//model->save("LBP_TrainData.xml");
  //  int predictedLabel = model->predict(testSample);
    //
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //


	
	char filename[100];
	int choose;
	vector<Mat> input;
	vector<int> labels;
	while(1)
	{
		printf("���������1Ϊ���������5Ϊʶ��������7Ϊ��������9�˳�\n");
		scanf("%d",&choose);
		if(9 == choose)
			break;
		if( 7 == choose)
		{
			model->save("../FaceData.xml");
			continue;
		}
		if(choose != 1 && choose != 5)
			continue;
		

		printf("�����ļ���\n");

		scanf("%s",filename);

		IplImage* img = cvLoadImage(filename);

		if(NULL == img)
		{
			printf("�޷�����Դͼ��\n");
			continue;
		}
		int resCount = 0;
		CvRect **rect = NULL;
		rect = displaydetection(img,resCount);

		if(!resCount || NULL == rect)
		{
			printf("��ⲻ������\n");
			cvReleaseImage(&img);
			continue;
		}
		IplImage *face = cvCreateImage(cvSize(FACE_WIDTH,FACE_HIGH),img->depth,img->nChannels);
		IplImage *face_gray = cvCreateImage(cvSize(FACE_WIDTH,FACE_HIGH),img->depth,1);
		
		printf("����%d����\n",resCount);
		cvNamedWindow("ShowFace",1);
		for(int i = 0;i < resCount;i++)
		{
			printf("��%d����\n",i);
			cvSetImageROI(img,*rect[i]);

			cvResize(img,face);

			cvCvtColor(face,face_gray,CV_RGB2GRAY);

			cvResetImageROI(img);

			cvShowImage("ShowFace",face);
			//cvWaitKey(0);

			//frcog.addFace(face_gray,p);
			if(5 == choose)
			{
				Mat face_mat(face_gray,false);
				

				int predict = model->predict(face_mat);

				cout << "Is " << predict << endl;
			}
			else if(1 == choose)
			{
				int label ;
				cout << "������" << endl;
				cin >> label;
				
				Mat face_mat(face_gray,false);
				input.push_back(face_mat);
				labels.push_back(label);
				model->update(input,labels);
			}
		}
		cvReleaseImage(&face);
		cvReleaseImage(&face_gray);
		cvReleaseImage(&img);
		input.clear();
		labels.clear();
		cvDestroyWindow("ShowFace");
		for(int i = 0;i < resCount;i++)
		{
			delete rect[i];
		}
		delete rect;
	}

    return 0;
}
