#include "common.h"
MYSQL dbobj;//handle
MYSQL_RES *pREs;//result
MYSQL_ROW sqlrow;//row

void mysql_test()
{
	int status;
	mysql_init(&dbobj);
	//if(!mysql_real_connect(&dbobj,"127.0.0.1:3306","rui","123","face",0,NULL,0))
	if (!mysql_real_connect(&dbobj, "localhost", "root", "456159", "face", 3306,
			NULL, CLIENT_FOUND_ROWS))
	{
		printf("connect mysql error\n");
		return ;
	}
	else
	{
		printf("success\n");
	}

	mysql_set_character_set(&dbobj, "utf-8");
	status =
			mysql_query(&dbobj,
					"insert into TrainData (nEigens,nTrainFaces,trainPersonID) values(1,2,3)");
	if (status != 0)
	{
		printf("query failure\n");
	}
	else
	{
		printf("query success:%d\n", status);
	}
}


int main()
{
    cout << "Hello world!" << endl;
	char a[6]={0};
    char b[10]={"abcdef"};

    my_strcpy(a,b,sizeof(a));
    printf("%s\n",a);
	//cout << a << endl;

	
    return 0;
}
