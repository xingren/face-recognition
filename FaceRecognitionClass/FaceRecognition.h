#ifndef FACERECOGNITION_H_INCLUDED
#define FACERECOGNITION_H_INCLUDED
#include "common.h"

void printError(const char *errstr);



class FileHandle
{
public:
	FILE *file;
	FileHandle(char filename[],char op[])
	{
		file = fopen(filename,op);
		if(NULL == file)
		{
			printf("can't not open the file: %s\n",filename);
			return ;
		}

	}
	//遇到空格、回车、EOF为正常结束,超过输出字符串最大长度为异常结束,bufState用以判断输出情况，0为输出大小超过maxSize还没正常结束，
	//1为不够长度写入字符串结束标志'\0\'，2为正常结束，-1为读文件到文件尾,-2为读到文件尾并且不够长度写入字符串结束标志'\0'
	bool readString(char output[],int maxSize,int *bufState  = NULL)
	{
		if(NULL == file)
		{
			printf("no file be open\n");
			return false;
		}
		char c = fgetc(file);
		while((c == ' ' || c == '\n' || c == '\t') && c != EOF)
			c = fgetc(file);

		if(c == EOF)
			return false;

		int index = 0;
		while(c != ' ' && c != ' ' && c != '\t' && c != EOF && index < maxSize)
		{
			output[index++]  = c;
			c = fgetc(file);
		}
		if(index == maxSize && EOF != c)
		{
			if(bufState != NULL)
				*bufState = 0;
			return true;
		}
		if(EOF == c)
		{
			//不够长度写入'\0'
			if(NULL != bufState && index + 1 >= maxSize)
				*bufState = -2;
			else if(NULL != bufState && index + 1 < maxSize)
				*bufState = -1,output[index+1] = '\0';
			return true;
		}
		output[index] = '\0';
	}
	int SetFileSeek(int offset,int orgin)
	{
		return fseek(file,offset,orgin);
	}
};

class PersonData
{
public:
	char *name;
	size_t nameSize;
	int sex;//0为女，1为男
	PersonData()
	{
		nameSize = 50;
		name = new char[nameSize];
		my_strcpy(name,INVALID_NAME,nameSize);
	}
	PersonData(const PersonData &p)
	{
		sex = p.sex;
		nameSize = p.nameSize;
		name = new char[nameSize];
		my_strcpy(name,p.name,nameSize);
	}
	PersonData(char n[],int s)
	{

		//添加错误处理
		int length = strlen(n);//防止strlen(n)出错。
		if(length > MAX_NAME_LENGTH)//防止length过大
		{
			printError("名字长度过长");
			return;
		}
		nameSize = length+1;
		name = new char[nameSize];//不能name = new char[strlen(n)+1] 因为如果n字符串数组没有‘\0'，这会导致一个错误的字符串长度
		sex = s;
		my_strcpy(name,n,nameSize);
	}
	~PersonData()
	{
		if(name != NULL)
			delete [] name;

	}
	bool changeName(char n[])
	{
		int length = strlen(n);
		if (length > MAX_NAME_LENGTH) //防止length过大
		{
			printError("名字长度过长");
			return false;
		}
		if(name != NULL)
		{

			if(nameSize > length)
				return my_strcpy(name,n,nameSize) == -1?false:true;
			else
			{
				nameSize = length + 1;
				delete [] name;
				name = new char[nameSize]; //不能name = new char[strlen(n)+1] 因为如果n字符串数组没有‘\0'，这会导致一个错误的字符串长度
				memset(name,0,nameSize*sizeof(char));
				return my_strcpy(name,n,nameSize) == -1?false:true;
			}

		}
		else
		{
			nameSize = length + 1;
			//delete [] name;
			name = new char[nameSize]; //不能name = new char[strlen(n)+1] 因为如果n字符串数组没有‘\0'，这会导致一个错误的字符串长度
			memset(name,0,nameSize*sizeof(char));
			return my_strcpy(name,n,nameSize) == -1?false:true;
		}
	}
	bool changeSex(int s)
	{
		if(s != 0 && s!=1)
		{
			printError("性别设置有误");
			return false;
		}
		sex = s;
		return true;
	}
	void writeToFile(FILE *file)
	{
		char str[4];
		fwrite(name,sizeof(char),strlen(name),file);
		sprintf(str," %d ",sex);
		fwrite(str,sizeof(char),strlen(str),file);
	}
	PersonData& operator=(const PersonData& p)
	{
		sex = p.sex;
		if(nameSize >= p.nameSize)
		{
			my_strcpy(name,p.name,nameSize);
		}
		else
		{
			if(NULL != name)
				delete [] name;
			nameSize = p.nameSize;
			name = new char[nameSize];
			my_strcpy(name,p.name,nameSize);


		}
		return *this;
	}

};

class FaceRecognition
{
private:
	class FaceData
	{
	public:
		PersonData person;
		float *eigenVec;
		size_t vecSize;
		FaceData()
		{
			eigenVec = NULL;

		}
		FaceData(float *e,size_t size,PersonData p)
		{
			person = p;
			this->vecSize = size;
			eigenVec = new float[size];
			if(eigenVec == NULL)
			{
				printError("memory not enough");


				//添加错误处理
				abort();
			}
			for(int i = 0;i < size;i++)
			{
				eigenVec[i] = e[i];
			}
		}
		FaceData(size_t size,PersonData p)
		{
			eigenVec = new float[size];
			this->vecSize = size;
			if(eigenVec == NULL)
			{
				printError("memory not enough");

				//添加错误处理
			//	abort();
			}
		}
		FaceData(const FaceData& fd)
		{
			person = fd.person;
			vecSize = fd.vecSize;
			this->eigenVec = new float[vecSize];
			memcpy(eigenVec,fd.eigenVec,sizeof(float)*vecSize);
		}
		~FaceData()
		{
			if(eigenVec != NULL)
				delete [] eigenVec;
		}
		bool setEigenVecSize(int n)
		{
			this->vecSize = n;
			eigenVec = new float[n];
			if(eigenVec == NULL)
			{
				printError("could not assign memory");
				//添加错误处理
			//	abort();
				return false;
			}
			return true;
		}
		bool setEigenVecSingleVal(float val,int index)
		{
			if(eigenVec == NULL)
			{
				printError("not be assigned memroy for eigenVec array");
				return false;
			}
			if(index > this->vecSize)
			{
				printError("index bigger than eigenVec array size");
				return false;
			}
			eigenVec[index] = val;
			return true;
		}
		FaceData&operator=(const FaceData& fd)
		{
			if(eigenVec == NULL && fd.eigenVec != NULL)
			{
				vecSize = fd.vecSize;
				eigenVec = new float[vecSize];
				memcpy(eigenVec,fd.eigenVec,sizeof(float)*vecSize);
			}
			else if(eigenVec != NULL && fd.eigenVec != NULL)
			{
				if(vecSize < fd.vecSize)
				{
					delete [] eigenVec;
					vecSize = fd.vecSize;
					eigenVec = new float[vecSize];

				}
				memcpy(eigenVec,fd.eigenVec,sizeof(float)*fd.vecSize);
			}
			return *this;
		}
	};
public:

	char dataFilePath[200];//人脸数据库文件所在的路径
	char cascadePath[100]; //人脸分类器路径
	char trainingDataPath[200];//训练结果路径
	char trainListPath[200];//训练所需的训练集的路径
	IplImage ** faceImgArr; // 指向训练人脸和测试人脸的指针（在学习和识别阶段指向不同）
	int faceImgArrSize;


	CvMat    * trainPersonIDMat; // 人脸图像的ID号
	int nTrainFaces ; // 训练图像的数目
	int nEigens  ; // 自己取的主要特征值数目
	IplImage * pAvgTrainImg; // 训练人脸数据的平均值
	IplImage ** eigenVectArr ; // 投影矩阵，也即主特征向量
	CvMat * trainEigenValMat       ; // 特征值
	CvMat * projectedTrainFaceMat; // 训练图像的投影
	CvMat * trainPersonMat;
	vector<FaceData> faceDataVec;//人脸数据库的特征值
	bool loadTrainData;//true为已经加载了训练库，false没有加载训练库



	int dataCount;
	FaceRecognition();

	~FaceRecognition();
	void learn();
	void recognize();
	void doPCA();
	void storeTrainingData();
	int  loadTrainingData(CvMat ** pTrainPersonNumMat);
	int  findNearestNeighbor(float * projectedTestFace);
	int  loadFaceImgArray(char * filename);
	void printUsage();
	//检测人脸
	CvRect** displaydetection(IplImage *pInpImg,int &resCount);

	//添加人脸至数据库,true为添加成功,false为添加不成功.注意：没有释放传入的IplImage*
	bool addFace(IplImage*,PersonData);

	//根据特征向量，查找是否在数据库中,第三个参数不为空则将对应的人的信息赋给PersonData变量
	int findPerson(const float* projectedAddFace,int len,PersonData* p = NULL);

	//保存一个人脸数据至
	bool saveFaceDataToFile(float *projectedAddFace,int num,PersonData,char filename[]=NULL );

	//加载本地人脸数据文件
	bool loadFaceDataFromFile(char filename[] = NULL);

	//获得本地人脸数据文件的数量
	int getFaceDataCountFromFile(FILE *file);

	//根据特征向量，检查是否在数据库中,第三个参数不为空则将对应的人的信息赋给PersonData变量
	bool checkInDataBase(float *eigenVec,int len,PersonData*);
	//返回在数据库的最大距离，当测试脸与数据库中对比时求得最近的距离，当大于maxDist时，则不在该数据库内
	double makeMaxDist();
	//最大距离
	double maxDist;
	PersonData recognize(IplImage*);
};


#endif // FACERECOGNITION_H_INCLUDED
