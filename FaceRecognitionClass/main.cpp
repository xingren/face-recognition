#include "common.h"
#include "FaceRecognition.h"
//MYSQL dbobj;//handle
//MYSQL_RES *pREs;//result
//MYSQL_ROW sqlrow;//row
//
//void mysql_test()
//{
//  int status;
//	mysql_init(&dbobj);
//	//if(!mysql_real_connect(&dbobj,"127.0.0.1:3306","rui","123","face",0,NULL,0))
//	if (!mysql_real_connect(&dbobj, "localhost", "root", "456159", "face", 3306,
//			NULL, CLIENT_FOUND_ROWS))
//	{
//		printf("connect mysql error\n");
//		return ;
//	}
//	else
//	{
//		printf("success\n");
//	}
//
//	mysql_set_character_set(&dbobj, "utf-8");
//	status =
//			mysql_query(&dbobj,
//					"insert into TrainData (nEigens,nTrainFaces,trainPersonID) values(1,2,3)");
//	if (status != 0)
//	{
//		printf("query failure\n");
//	}
//	else
//	{
//		printf("query success:%d\n", status);
//	}
//}

FaceRecognition frcog;
int test()
{
	PersonData p("jack",0);
	char filename[100];
	int choose;
	while(1)
	{
		printf("输入操作，1为添加人脸，5为识别人脸，9退出\n");
		scanf("%d",&choose);
		if(9 == choose)
			break;
		if(choose != 1 && choose != 5)
			continue;
		printf("输入文件名\n");

		scanf("%s",filename);

		IplImage* img = cvLoadImage(filename);

		if(NULL == img)
		{
			printf("无法加载源图像\n");
			continue;
		}
		int resCount = 0;
		CvRect **rect = NULL;
		rect = frcog.displaydetection(img,resCount);

		if(!resCount || NULL == rect)
		{
			printf("检测不出人脸\n");
			cvReleaseImage(&img);
			continue;
		}
		IplImage *face = cvCreateImage(cvSize(FACE_WIDTH,FACE_HIGH),img->depth,img->nChannels);
		IplImage *face_gray = cvCreateImage(cvSize(FACE_WIDTH,FACE_HIGH),img->depth,1);
		
		printf("检测出%d张脸\n",resCount);
		cvNamedWindow("ShowFace",1);
		for(int i = 0;i < resCount;i++)
		{
			printf("第%d张脸\n",i);
			cvSetImageROI(img,*rect[i]);

			cvResize(img,face);

			cvCvtColor(face,face_gray,CV_RGB2GRAY);

			cvResetImageROI(img);

			cvShowImage("ShowFace",face);
			cvWaitKey(0);

			//frcog.addFace(face_gray,p);
			if(5 == choose)
			{
				PersonData rp = frcog.recognize(face_gray);
				if(0 == strcmp(rp.name,INVALID_NAME))
				{
					printf("无法识别\n");

				}
				else
				{
					printf("识别成功，姓名为%s，性别为%d\n",rp.name,rp.sex);
				}
			}
			else
			{
				char name[50];
				int sex;
				printf("输入姓名性别，空格隔开\n");
				scanf("%s%d",name,&sex);

				PersonData tp(name,sex);
				frcog.addFace(face_gray,tp);
			}
		}
		cvReleaseImage(&face);
		cvReleaseImage(&face_gray);
		cvReleaseImage(&img);
		cvDestroyWindow("ShowFace");
		for(int i = 0;i < resCount;i++)
		{
			delete rect[i];
		}
		delete rect;
	}
	return 0;
}

int main()
{

	//	frcog.learn();
	//	frcog.recognize();
	//	return 0;
	//	frcog.loadFaceDataFromFile();

	test();

	return 0;
}
