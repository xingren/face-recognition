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

		//将捕捉的脸保存
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

//学习阶段代码
void learn()
{
	int i, offset;

	//加载训练图像集
	nTrainFaces = loadFaceImgArray("train.txt");
	if( nTrainFaces < 2 )
	{
		fprintf(stderr,
			"Need 2 or more training faces\n"
			"Input file contains only %d\n", nTrainFaces);
		return;
	}

	// 进行主成分分析
	doPCA();

	//将训练图集投影到子空间中
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

	//将训练阶段得到的特征值，投影矩阵等数据存为.xml文件，以备测试时使用
	storeTrainingData();
}

//识别阶段代码
void recognize()
{
	int i, nTestFaces  = 0;         // 测试人脸数
	CvMat * trainPersonNumMat = 0;  // 训练阶段的人脸数
	float * projectedTestFace = 0;

	// 加载测试图像，并返回测试人脸数
	nTestFaces = loadFaceImgArray("test.txt");
	printf("%d test faces loaded\n", nTestFaces);

	// 加载保存在.xml文件中的训练结果
	if( !loadTrainingData( &trainPersonNumMat ) ) return;

	// 
	projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );
	for(i=0; i<nTestFaces; i++)
	{
		int iNearest, nearest, truth;

		//将测试图像投影到子空间中
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

//加载保存过的训练结果
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

//存储训练结果
void storeTrainingData()
{
	CvFileStorage * fileStorage;
	int i;

	
	fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );

	//存储特征值，投影矩阵，平均矩阵等训练结果
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


//寻找最接近的图像
int findNearestNeighbor(float * projectedTestFace)
{

	double leastDistSq = DBL_MAX;		//定义最小距离，并初始化为无穷大
	int i, iTrain, iNearest = 0;

	for(iTrain=0; iTrain<nTrainFaces; iTrain++)
	{
		double distSq=0;

		for(i=0; i<nEigens; i++)
		{
			float d_i =
				projectedTestFace[i] -
				projectedTrainFaceMat->data.fl[iTrain*nEigens + i];
		distSq += d_i*d_i / eigenValMat->data.fl[i];  // Mahalanobis算法计算的距离
		//	distSq += d_i*d_i; // Euclidean算法计算的距离
		}

		if(distSq < leastDistSq)
		{
			leastDistSq = distSq;
			iNearest = iTrain;
		}
	}

	return iNearest;
}

//主成分分析
void doPCA()
{
	int i;
	CvTermCriteria calcLimit;
	CvSize faceImgSize;

	// 自己设置主特征值个数
	nEigens = nTrainFaces-1;

	//分配特征向量存储空间
	faceImgSize.width  = faceImgArr[0]->width;
	faceImgSize.height = faceImgArr[0]->height;
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);	//分配个数为住特征值个数
	for(i=0; i<nEigens; i++)
		eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	//分配主特征值存储空间
	eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

	// 分配平均图像存储空间
	pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// 设定PCA分析结束条件
	calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);

	// 计算平均图像，特征值，特征向量
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

//加载txt文件的列举的图像
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

	// 统计人脸数
	while( fgets(imgFilename, 512, imgListFile) ) ++nFaces;
	rewind(imgListFile);

	// 分配人脸图像存储空间和人脸ID号存储空间
	faceImgArr        = (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
	personNumTruthMat = cvCreateMat( 1, nFaces, CV_32SC1 );

	for(iFace=0; iFace<nFaces; iFace++)
	{
		// 从文件中读取序号和人脸名称
		fscanf(imgListFile,
			"%d %s", personNumTruthMat->data.i+iFace, imgFilename);

		// 加载人脸图像
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
	CvHaarClassifierCascade* pCascade=0;		//指向后面从文件中获取的分类器
	CvMemStorage* pStorage=0;					//存储检测到的人脸数据
	CvSeq* pFaceRectSeq;			
	pStorage=cvCreateMemStorage(0);				//创建默认大先64k的动态内存区域
	pCascade=(CvHaarClassifierCascade*)cvLoad("d:/tools/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml");		//加载分类器
	cvLoadImage(filename);
	if (!pInpImg||!pStorage||!pCascade)
	{
		printf("initialization failed:%s\n",(!pInpImg)?"can't load image file":(!pCascade)?"can't load haar-cascade---make sure path is correct":"unable to allocate memory for data storage",argv[1]);
		return ;
	}
	//人脸检测
	pFaceRectSeq=cvHaarDetectObjects(pInpImg,pCascade,pStorage,
		1.2,2,CV_HAAR_DO_CANNY_PRUNING,cvSize(40,40));
	//将检测到的人脸以矩形框标出。
	IplImage *reuslt;
	displaydetection(pInpImg,pFaceRectSeq,filename,reuslt);




	cvReleaseImage(&pInpImg);
	cvReleaseHaarClassifierCascade(&pCascade);
	cvReleaseMemStorage(&pStorage);
}