#include <iostream>
#include <string>

struct X
{
	//X() {}
	~X() {};
	
	int a;
	int b;
};

union UX
{
	//UX() {}
	~UX() {} ;
	int a;
	X	x;
};

class A
{
public:
	//A()	{ new (&ux.x) X(); }
	A() {}
	~A() {}
	
	UX ux;
};

int main()
{
	A a;
	return  0;
}
