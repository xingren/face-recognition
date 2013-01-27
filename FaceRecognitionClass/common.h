#ifndef COMMON_H_INCLUDED
#define COMMON_H_INCLUDED

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>



#ifdef WIN32
#include <Windows.h>
#include <mysql.h>


#elif _linux_
#include <mysql/mysql.h>



#endif
//some constant value

using namespace std;

#define MAX_NAME_LENGTH 100
int my_strcpy(char dest[],const char s[],size_t size);


#endif // COMMON_H_INCLUDED
