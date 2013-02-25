#ifndef COMMON_H_INCLUDED
#define COMMON_H_INCLUDED


#ifdef WIN32
//#include <Windows.h>
//#include <mysql.h>


#elif _linux_
#include <mysql/mysql.h>



#endif


#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv/cvaux.h>





//some constant value

using namespace std;

#define MAX_NAME_LENGTH 100
int my_strcpy(char dest[],const char s[],size_t size);

#define FACE_WIDTH 92
#define FACE_HIGH 112
#define INVALID_NAME "invalid name"


#endif // COMMON_H_INCLUDED
