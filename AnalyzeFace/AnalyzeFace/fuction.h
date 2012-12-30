#ifndef FUCTION
#define FUCTION
#include"stdafx.h"


void learn();
void recognize();
void doPCA();
void storeTrainingData();
int  loadTrainingData(CvMat ** pTrainPersonNumMat);
int  findNearestNeighbor(float * projectedTestFace);
int  loadFaceImgArray(char * filename);
void printUsage();
void displaydetection(IplImage* pInpImg,CvSeq* pFaceRectSeq,char* FileName,IplImage *result);

#endif