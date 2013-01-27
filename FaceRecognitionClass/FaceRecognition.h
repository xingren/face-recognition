#ifndef FACERECOGNITION_H_INCLUDED
#define FACERECOGNITION_H_INCLUDED
#include "common.h"

void printError(char *errstr);

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
		my_strcpy(name,"no name",nameSize);
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

	}
	~PersonData()
	{
		if(name != NULL)
			delete name;

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
				return my_strcpy(name,n,nameSize);
			else
			{
				nameSize = length + 1;
				delete name;
				name = new char[nameSize]; //不能name = new char[strlen(n)+1] 因为如果n字符串数组没有‘\0'，这会导致一个错误的字符串长度
				memset(name,0,nameSize*sizeof(char));
				return my_strcpy(name,n,nameSize);
			}

		}
		else
		{
			nameSize = length + 1;
			//delete name;
			name = new char[nameSize]; //不能name = new char[strlen(n)+1] 因为如果n字符串数组没有‘\0'，这会导致一个错误的字符串长度
			memset(name,0,nameSize*sizeof(char));
			return my_strcpy(name,n,nameSize);
		}
	}
	bool changeSex(int s)
	{
		if(s != 0 || s!=1)
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
		~FaceData()
		{
			if(eigenVec != NULL)
				delete eigenVec;
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
	};
public:
	
	char dataFilePath[200];//人脸数据库文件所在的路径

	IplImage ** faceImgArr; // 指向训练人脸和测试人脸的指针（在学习和识别阶段指向不同）
	CvMat    * trainPersonIDMat; // 人脸图像的ID号
	int nTrainFaces ; // 训练图像的数目
	int nEigens  ; // 自己取的主要特征值数目
	IplImage * pAvgTrainImg; // 训练人脸数据的平均值
	IplImage ** eigenVectArr ; // 投影矩阵，也即主特征向量
	CvMat * trainEigenValMat       ; // 特征值
	CvMat * projectedTrainFaceMat; // 训练图像的投影
	vector<FaceData> faceDataVec;//人脸数据库的特征值
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
	CvRect* displaydetection(IplImage *pInpImg,int &resCount);//检测人脸
	bool addFace(IplImage*,PersonData);//添加人脸至数据库,true为添加成功,false为添加不成功.注意：没有释放传入的IplImage*
	int findPerson(float* projectedAddFace,int len,PersonData*);//根据特征向量，查找是否在数据库中,第三个参数不为空则将对应的人的信息赋给PersonData变量
	bool saveFaceDataToFile(float *projectedAddFace,int num,PersonData,char *filename );//保存一个人脸数据至
	bool loadFaceDataFromFile(char filename[]);//加载本地人脸数据文件
	int getFaceDataCountFromFile(FILE *file);//获得本地人脸数据文件的数量
	bool checkInDataBase(float *eigenVec,int len,PersonData*);//根据特征向量，检查是否在数据库中,第三个参数不为空则将对应的人的信息赋给PersonData变量

};


#endif // FACERECOGNITION_H_INCLUDED
