#include "common.h"
#include "FaceRecognition.h"
void printError(const char errstr[])
{
    printf("%s\n",errstr);
}

//size为dest数组的大小
int my_strcpy(char dest[],const char s[],size_t size)//size为dest数组的大小
{
	int len = strlen(s);

	if(size <= len)
		return -1;


	//添加错误处理
	/***
	 * dest和s的地址可能会重叠
	 */


	memcpy(dest,s,len*sizeof(char));
	dest[len] = 0;

	return 0;
}
