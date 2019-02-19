#include <iostream>
#include <string.h>
#include <sstream>
#include <stdlib.h>



unsigned int converIP(const char* ip)
{
	char szIP[32];
	strcpy(szIP, ip);
	unsigned int  dwIP = 0;
	
	char* p1 = szIP;
	char* p2 = strchr(szIP, '.');
	*p2 = 0;
	int d4 = atol(p1);

	p1 = ++p2;
	p2 = strchr(p2, '.');
	*p2 = 0;
	int d3 = atol(p1);

	p1 = ++p2;
	p2 = strchr(p2, '.');
	*p2 = 0;
	int d2 = atol(p1);


	p1 = ++p2;
	int d1 = atol(p1);

	dwIP = (d4 << 24) & 0xff000000 | (d3 <<16) & 0x00ff0000 | (d2 << 8) & 0x0000ff00 | (d1 & 0x000000ff);


std::cout << d4 << " " << d3 << " " << d2 << " " << d1 << std::endl;

	return dwIP;
}


int main()
{
	//std::cout << converIP("192.168.231.2") << std::endl;
    {
        uint64_t id = 12195900;
        //id ^= 0x1A2B3C;
        std::ostringstream ostm{};
        ostm << std::oct << id;
        auto s = ostm.str();
        std::cout << s << "\n";
        auto v = atoi(s.c_str());
        std::cout << (v^0x1A2B3C) << "\n";
    }
	return 0;
}



