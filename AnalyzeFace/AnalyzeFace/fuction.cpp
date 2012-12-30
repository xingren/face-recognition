#include "stdafx.h"

void displaydetection(IplImage* pInpImg,CvSeq* pFaceRectSeq,char* FileName,IplImage *result)
{
	int i;
	cvNamedWindow("haar window",1);
	printf("the number of face is %d",pFaceRectSeq->total);
	for (i=0;i<(pFaceRectSeq?pFaceRectSeq->total:0);i++)
	{
		CvRect* r=(CvRect*)cvGetSeqElem(pFaceRectSeq,i);
		CvPoint pt1={r->x,r->y};
		CvPoint pt2={r->x+r->width,r->y+r->height};

		cvSetImageROI(pInpImg,*r);

		//����׽��������
		result=cvCreateImage(cvSize(92,112),pInpImg->depth,pInpImg->nChannels);
		cvResize(pInpImg,result,CV_INTER_LINEAR);



	//	cvSaveImage("lena1.jpg",result);
		cvResetImageROI(pInpImg);
		cvRectangle(pInpImg,pt1,pt2,CV_RGB(0,255,0),3,4,0);
	}
	
	cvShowImage("haar window",pInpImg);
//	cvResetImageROI(pInpImg);
	cvWaitKey(0);
	cvDestroyWindow("haar window");
}

//ѧϰ�׶δ���
void learn()
{
	int i, offset;

	//����ѵ��ͼ��
	nTrainFaces = loadFaceImgArray("train.txt");
	if( nTrainFaces < 2 )
	{
		fprintf(stderr,
			"Need 2 or more training faces\n"
			"Input file contains only %d\n", nTrainFaces);
		return;
	}

	// �������ɷַ���
	doPCA();

	//��ѵ��ͼ��ͶӰ���ӿռ���
	projectedTrainFaceMat = cvCreateMat( nTrainFaces, nEigens, CV_32FC1 );
	offset = projectedTrainFaceMat->step / sizeof(float);
	for(i=0; i<nTrainFaces; i++)
	{
		//int offset = i * nEigens;
		cvEigenDecomposite(
			faceImgArr[i],
			nEigens,
			eigenVectArr,
			0, 0,
			pAvgTrainImg,
			//projectedTrainFaceMat->data.fl + i*nEigens);
			projectedTrainFaceMat->data.fl + i*offset);
	}

	//��ѵ���׶εõ�������ֵ��ͶӰ��������ݴ�Ϊ.xml�ļ����Ա�����ʱʹ��
	storeTrainingData();
}

//ʶ��׶δ���
void recognize()
{
	int i, nTestFaces  = 0;         // ����������
	CvMat * trainPersonNumMat = 0;  // ѵ���׶ε�������
	float * projectedTestFace = 0;

	// ���ز���ͼ�񣬲����ز���������
	nTestFaces = loadFaceImgArray("test.txt");
	printf("%d test faces loaded\n", nTestFaces);

	// ���ر�����.xml�ļ��е�ѵ�����
	if( !loadTrainingData( &trainPersonNumMat ) ) return;

	// 
	projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );
	for(i=0; i<nTestFaces; i++)
	{
		int iNearest, nearest, truth;

		//������ͼ��ͶӰ���ӿռ���
		cvEigenDecomposite(
			faceImgArr[i],
			nEigens,
			eigenVectArr,
			0, 0,
			pAvgTrainImg,
			projectedTestFace);

		iNearest = findNearestNeighbor(projectedTestFace);
		truth    = personNumTruthMat->data.i[i];
		nearest  = trainPersonNumMat->data.i[iNearest];

		printf("nearest = %d, Truth = %d\n", nearest, truth);
	}
}

//���ر������ѵ�����
int loadTrainingData(CvMat ** pTrainPersonNumMat)
{
	CvFileStorage * fileStorage;
	int i;

	
	fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
	if( !fileStorage )
	{
		fprintf(stderr, "Can't open facedata.xml\n");
		return 0;
	}

	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	eigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
	pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
	eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}

	
	cvReleaseFileStorage( &fileStorage );

	return 1;
}

//�洢ѵ�����
void storeTrainingData()
{
	CvFileStorage * fileStorage;
	int i;

	
	fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );

	//�洢����ֵ��ͶӰ����ƽ�������ѵ�����
	cvWriteInt( fileStorage, "nEigens", nEigens );
	cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
	cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0,0));
	cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
	cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
	cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));
	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
	}


	cvReleaseFileStorage( &fileStorage );
}


//Ѱ����ӽ���ͼ��
int findNearestNeighbor(float * projectedTestFace)
{

	double leastDistSq = DBL_MAX;		//������С���룬����ʼ��Ϊ�����
	int i, iTrain, iNearest = 0;

	for(iTrain=0; iTrain<nTrainFaces; iTrain++)
	{
		double distSq=0;

		for(i=0; i<nEigens; i++)
		{
			float d_i =
				projectedTestFace[i] -
				projectedTrainFaceMat->data.fl[iTrain*nEigens + i];
		distSq += d_i*d_i / eigenValMat->data.fl[i];  // Mahalanobis�㷨����ľ���
		//	distSq += d_i*d_i; // Euclidean�㷨����ľ���
		}

		if(distSq < leastDistSq)
		{
			leastDistSq = distSq;
			iNearest = iTrain;
		}
	}

	return iNearest;
}

//���ɷַ���
void doPCA()
{
	int i;
	CvTermCriteria calcLimit;
	CvSize faceImgSize;

	// �Լ�����������ֵ����
	nEigens = nTrainFaces-1;

	//�������������洢�ռ�
	faceImgSize.width  = faceImgArr[0]->width;
	faceImgSize.height = faceImgArr[0]->height;
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);	//�������Ϊס����ֵ����
	for(i=0; i<nEigens; i++)
		eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	//����������ֵ�洢�ռ�
	eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

	// ����ƽ��ͼ��洢�ռ�
	pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// �趨PCA������������
	calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);

	// ����ƽ��ͼ������ֵ����������
	cvCalcEigenObjects(
		nTrainFaces,
		(void*)faceImgArr,
		(void*)eigenVectArr,
		CV_EIGOBJ_NO_CALLBACK,
		0,
		0,
		&calcLimit,
		pAvgTrainImg,
		eigenValMat->data.fl);

	cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}

//����txt�ļ����оٵ�ͼ��
int loadFaceImgArray(char * filename)
{
	FILE * imgListFile = 0;
	char imgFilename[512];
	int iFace, nFaces=0;


	if( !(imgListFile = fopen(filename, "r")) )
	{
		fprintf(stderr, "Can\'t open file %s\n", filename);
		return 0;
	}

	// ͳ��������
	while( fgets(imgFilename, 512, imgListFile) ) ++nFaces;
	rewind(imgListFile);

	// ��������ͼ��洢�ռ������ID�Ŵ洢�ռ�
	faceImgArr        = (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
	personNumTruthMat = cvCreateMat( 1, nFaces, CV_32SC1 );

	for(iFace=0; iFace<nFaces; iFace++)
	{
		// ���ļ��ж�ȡ��ź���������
		fscanf(imgListFile,
			"%d %s", personNumTruthMat->data.i+iFace, imgFilename);

		// ��������ͼ��
		faceImgArr[iFace] = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);

		if( !faceImgArr[iFace] )
		{
			fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
			return 0;
		}
	}

	fclose(imgListFile);

	return nFaces;
}
void addFace(char *filename,char *personName)
{
	IplImage* pInpImg=0;
	CvHaarClassifierCascade* pCascade=0;		//ָ�������ļ��л�ȡ�ķ�����
	CvMemStorage* pStorage=0;					//�洢��⵽����������
	CvSeq* pFaceRectSeq;			
	pStorage=cvCreateMemStorage(0);				//����Ĭ�ϴ���64k�Ķ�̬�ڴ�����
	pCascade=(CvHaarClassifierCascade*)cvLoad("d:/tools/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml");		//���ط�����
	cvLoadImage(filename);
	if (!pInpImg||!pStorage||!pCascade)
	{
		printf("initialization failed:%s\n",(!pInpImg)?"can't load image file":(!pCascade)?"can't load haar-cascade---make sure path is correct":"unable to allocate memory for data storage",argv[1]);
		return ;
	}
	//�������
	pFaceRectSeq=cvHaarDetectObjects(pInpImg,pCascade,pStorage,
		1.2,2,CV_HAAR_DO_CANNY_PRUNING,cvSize(40,40));
	//����⵽�������Ծ��ο�����
	IplImage *reuslt;
	displaydetection(pInpImg,pFaceRectSeq,filename,reuslt);




	cvReleaseImage(&pInpImg);
	cvReleaseHaarClassifierCascade(&pCascade);
	cvReleaseMemStorage(&pStorage);
}