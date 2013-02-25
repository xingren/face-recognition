
#include "FaceRecognition.h"
FaceRecognition::FaceRecognition()
{
	dataCount = 0;
	loadTrainData = false;
	strcpy(dataFilePath,"../FaceData.dat");
	maxDist = DBL_MIN;
    strcpy(cascadePath,"../haarcascade_frontalface_alt2.xml");
	strcpy(trainingDataPath,"../TrainResult.xml");
	strcpy(trainListPath,"../train.txt");

}

FaceRecognition::~FaceRecognition()
{
	cvReleaseMat(&trainPersonIDMat);
	cvReleaseMat(&trainEigenValMat);
	cvReleaseMat(&projectedTrainFaceMat);
	cvReleaseMat(&trainPersonMat);
	faceDataVec.clear();
	if(faceImgArr)
	{
		for(int i = 0;i < faceImgArrSize;i++)
			cvReleaseImage(&faceImgArr[i]);
		delete [] faceImgArr;
	}
}
CvRect** FaceRecognition::displaydetection(IplImage *pInpImg,int &resCount)
{

	CvHaarClassifierCascade* pCascade=0;		//指向后面从文件中获取的分类器
	CvMemStorage* pStorage=0;					//存储检测到的人脸数据
	CvSeq* pFaceRectSeq;
	pStorage=cvCreateMemStorage(0);				//创建默认大先64k的动态内存区域

	//需要根据实际情况修改:加载分类器的路径
	pCascade=(CvHaarClassifierCascade*)cvLoad(cascadePath);		//加载分类器
	if (!pInpImg||!pStorage||!pCascade)
	{
		printf("initialization failed:%s\n",(!pInpImg)?"can't load image file":(!pCascade)?"can't load haar-cascade---make sure path is correct":"unable to allocate memory for data storage","pass a wrong IplImage*");
		return NULL;
	}
	//人脸检测
	pFaceRectSeq=cvHaarDetectObjects(pInpImg,pCascade,pStorage,
		1.2,2,CV_HAAR_DO_CANNY_PRUNING,cvSize(40,40));
	//将检测到的人脸以矩形框标出。
	int i;
	cvNamedWindow("haar window",1);
	printf("the number of face is %d\n",pFaceRectSeq->total);
	if(pFaceRectSeq->total == 0)
		return NULL;

	IplImage *result = cvCreateImage(cvSize(FACE_WIDTH,FACE_HIGH),pInpImg->depth,pInpImg->nChannels);
	resCount = pFaceRectSeq->total;
	//if(NULL == result)
	//	//添加错误处理：无法分配内存
	//{}


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

		//将捕捉的脸保存


		//cvSetImageROI(pInpImg,*r);
		cvResize(pInpImg,result,CV_INTER_LINEAR);
		sprintf(str,"%d",i);
		strcat(str,".jpg");
		cvSaveImage(str,result);

		cvResetImageROI(pInpImg);
		//cvResetImageROI(pInpImg);
		cvRectangle(pInpImg,pt1,pt2,CV_RGB(0,255,0),3,4,0);

	}

	cvShowImage("haar window",pInpImg);
	//	cvResetImageROI(pInpImg);
	cvWaitKey(0);
	cvDestroyWindow("haar window");
	//cvReleaseImage(&pInpImg);
	cvReleaseHaarClassifierCascade(&pCascade);
	cvReleaseMemStorage(&pStorage);
	cvReleaseImage(&result);

	return res;
}

void FaceRecognition::learn()
{
	int i, offset;

	//加载训练图像集
	nTrainFaces = loadFaceImgArray(trainListPath);
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



void FaceRecognition::doPCA()
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
	trainEigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

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
		trainEigenValMat->data.fl);

	cvNormalize(trainEigenValMat, trainEigenValMat, 1, 0, CV_L1, 0);
}

int FaceRecognition::findNearestNeighbor(float * projectedTestFace)
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
			distSq += d_i*d_i / trainEigenValMat->data.fl[i];  // Mahalanobis算法计算的距离
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

int FaceRecognition::loadFaceImgArray(char * filename)
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
	//	fseek(imgListFile,1,SEEK_SET);
	// 分配人脸图像存储空间和人脸ID号存储空间
	faceImgArr        = (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
	faceImgArrSize = nFaces;

	trainPersonIDMat = cvCreateMat( 1, nFaces, CV_32SC1 );
	for(iFace=0; iFace<nFaces; iFace++)
	{
		// 从文件中读取序号和人脸名称
		fscanf(imgListFile,
			"%s %d",  imgFilename,(trainPersonIDMat)->data.i+iFace);

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

int FaceRecognition::loadTrainingData(CvMat ** pTrainPersonNumMat)
{
	CvFileStorage * fileStorage;
	int i;


	fileStorage = cvOpenFileStorage( trainingDataPath, 0, CV_STORAGE_READ );
	if( !fileStorage )
	{
		fprintf(stderr, "Can't open %s\n",trainingDataPath);
		return 0;
	}

	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	trainEigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "trainEigenValMat", 0);
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
	loadTrainData = true;
	return 1;
}

void FaceRecognition::recognize()
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
		truth    = trainPersonIDMat->data.i[i];
		nearest  = trainPersonNumMat->data.i[iNearest];

		printf("nearest = %d, Truth = %d\n", nearest, truth);
	}
}

void FaceRecognition::storeTrainingData()
{
	CvFileStorage * fileStorage;
	int i;


	fileStorage = cvOpenFileStorage( trainingDataPath, 0, CV_STORAGE_WRITE );

	//存储特征值，投影矩阵，平均矩阵等训练结果
	cvWriteInt( fileStorage, "nEigens", nEigens );
	cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
	cvWrite(fileStorage, "trainPersonNumMat", trainPersonIDMat, cvAttrList(0,0));
	cvWrite(fileStorage, "trainEigenValMat", trainEigenValMat, cvAttrList(0,0));
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

bool FaceRecognition::checkInDataBase(float *eigenVec,int len,PersonData* person = NULL)
{



	return true;
}

bool FaceRecognition::addFace(IplImage* faceImg,PersonData person)
{
	//计算特征徝

	float* projectedAddFace;
	CvMat *trainingPersonID;

	printf("start cal the new face eigen val\n");


	if(!loadTrainData)
		loadTrainingData(&trainingPersonID),loadTrainData = true;
	projectedAddFace = new float[nEigens];
	cvEigenDecomposite(faceImg,nEigens,eigenVectArr,0,0,pAvgTrainImg,projectedAddFace);

	//先判断增加的人在不在数据库中
	int iNearest = findPerson(projectedAddFace,nEigens);


	//不在数据库中
	if(-1 == iNearest)
	{
		if(!saveFaceDataToFile(projectedAddFace,nEigens,person))
		{
			printError("写入数据库失败");
			//添加错误处理
			return false;
		}
		//将添加的人脸添加至faceDataVec
		faceDataVec.push_back(FaceData(projectedAddFace,nEigens,person));
	}
	else
	{
		printf("欲添加的人脸已经在数据库中，名字为%s\n",faceDataVec[iNearest].person.name);
		return false;
	}


	//delete [] projectedAddFace;
	delete [] projectedAddFace;
	return true;
}


PersonData FaceRecognition::recognize(IplImage* img)
{
	float* projectedVal;
	CvMat* trainingPersonID;
	

	if(!loadTrainData)
		loadTrainingData(&trainingPersonID),loadTrainData = true;
    
	projectedVal = new float[nEigens];
	cvEigenDecomposite(img,nEigens,eigenVectArr,NULL,NULL,pAvgTrainImg,projectedVal);
     
	PersonData person;

	

	findPerson(projectedVal,nEigens,&person);

	printf("projectedVal point addr:%ld\n",projectedVal);
	delete [] projectedVal;
	return person;

}

bool FaceRecognition::saveFaceDataToFile(float *projectedAddFace,int num,PersonData person,char *filename)
{
	FILE *dataFile;
	if(filename == NULL)
		filename = dataFilePath;
	dataFile = fopen(filename, "w+");
	if (dataFile == NULL)
	{
		printError("can't open faceData.dat");
		return false;
	}
	char str[200];


	char c;

	//获得有FaceData.dat有多少数据
	int count = 0,index = 0;
	count = getFaceDataCountFromFile(dataFile);
	int tmp = count;
	
	//重定位文件流的指针,覆盖掉原来文件末尾的数据总数
	while(tmp)
		tmp /= 10,index--;

	fseek(dataFile,index,SEEK_END);

	printf("start to write to FaceData.dat\n");

	//写入个人信息
	person.writeToFile(dataFile);
	fwrite(" ",sizeof(char),strlen(" "),dataFile);

	//写入特征个数
	//潜在BUG，strlen(str)
	sprintf(str,"%d ",num);
	fwrite(str,sizeof(char),strlen(str),dataFile);



	size_t tsize;
	for(int i = 0;i < num;i++)
	{
		//潜在BUG，strlen(str)
		sprintf(str,"%lf ",projectedAddFace[i]);
		tsize = fwrite(str,sizeof(char),strlen(str),dataFile);
		if(tsize != strlen(str))
		{
			printError("write to FaceData error");
			//添加错误处理
			/***
			*
			*/
		}
	}
	//写入数据库的总数
	count++;
	sprintf(str,"%d",count);

	//潜在BUG，strlen(str)
	fwrite(str,sizeof(char),strlen(str),dataFile);
	fflush(dataFile);
	fclose(dataFile);

	printf("finish to write to FaceData.dat\n");

	return true;

}

double FaceRecognition::makeMaxDist()
{
	double maxDist = DBL_MIN;

	CvMat *personID;
	if(!loadTrainData)
		loadTrainingData(&personID),loadTrainData = true;

	double d,distSq;
	for(int i = 0;i < nTrainFaces;i++)
	{
		for(int j = i+1;j < nTrainFaces;j++)
		{
			distSq = 0;
			for(int k = 0;k < nEigens;k++)
			{
				d = projectedTrainFaceMat->data.fl[i*nEigens + k] - projectedTrainFaceMat->data.fl[j*nEigens + k];
				distSq += d*d;
			}
			if(distSq > maxDist)
				maxDist = distSq;
		}
	}
	return maxDist;

}

int FaceRecognition::findPerson(const float* projectedAddFace,int len,PersonData* person)
{
	int iNearest = -1;
	double leastDist = DBL_MAX;
	if(0 == faceDataVec.size())
	{
		loadFaceDataFromFile(dataFilePath);
	}

	double tval;



	double tmp = fabs(maxDist - DBL_MIN);

	CvMat *personID;

	if(tmp >= 0 && tmp <= DBL_MIN)
		maxDist = makeMaxDist()/2.5;

	double d,distSq;
	for(int i = 0;i < faceDataVec.size();i++)
	{
		distSq = 0;
		for(int j = 0;j < faceDataVec.at(i).vecSize && j < len;j++)
		{
			d = projectedAddFace[j] - faceDataVec.at(i).eigenVec[j];
			distSq += d*d;
		}
		if(leastDist > distSq)
		{
			leastDist = distSq;
			iNearest = i;
		}
	}
	if(leastDist > maxDist)
		iNearest = -1;
	if(-1 != iNearest && NULL != person)
	{
		(*person) = faceDataVec.at(iNearest).person;
	}
	return iNearest;
}



bool FaceRecognition::loadFaceDataFromFile(char *filename)
{
	if(NULL == filename)
		filename = dataFilePath;
	FILE *dataFile;
	dataFile = fopen(filename, "r");
	if (NULL == dataFile)
	{
		printError("can't open faceData.dat");
		return false;
	}
	char str[MAX_NAME_LENGTH];

	int count = getFaceDataCountFromFile(dataFile);

	if(0 != faceDataVec.size())
	{
		faceDataVec.clear();
	}


	FaceData tmp;
	int sex;
	int n;
	float val;
	char c;
	int index;
	for(int i = 0;i < count;i++)
	{
		//添加错误处理：当文件格式不对时，fscanf函数会出问题
		//人物信息
		fscanf(dataFile,"%s",str);
		tmp.person.changeName(str);
		fscanf(dataFile,"%d",&sex);
		tmp.person.changeSex(sex);

		//特征值个数
		fscanf(dataFile,"%d",&n);


		if(! tmp.setEigenVecSize(n))
		{
			//添加错误处理
		}
		for(int j = 0;j < n;j++)
		{
			fscanf(dataFile,"%f",&val);
			if(!tmp.setEigenVecSingleVal(val,j))
			{
				//添加错误处理
			}
		}
		faceDataVec.push_back(tmp);
	}

	fclose(dataFile);
	return true;
}



int FaceRecognition::getFaceDataCountFromFile(FILE *file)
{

	if(file == NULL)
		return 0;
	char c;
	//获得有FaceData.dat有多少数据
	int count = 0,index = -1;

	//添加错误处理，offset可能为-1
	int offset = ftell(file);
	fseek(file,index,SEEK_END);
	c = fgetc(file);
	while( (c >='0' && c<='9') && c != EOF)
	{
		count = count*10+c - '0';
		index--;
		fseek(file,index,SEEK_END);
		c = fgetc(file);

	}

	//添加错误处理，该函数执行可能失败
	fseek(file,offset,SEEK_SET);


	return count;

}
