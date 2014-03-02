
INCLUDE:= 	-I./

CPPFLAGS:= -g -Wall -std=c++11

OBJS:=	main.o

all: clean main

test:
	g++ $(CPPFLAGS) test.cpp $(INCLUDE)
	
main:$(OBJS)
	g++ $(CPPFLAGS) $^ $(INCLUDE) -o $@


.cpp.o:
	g++ $(CPPFLAGS) $^ $(INCLUDE) -c
	
.PHONY:clean
clean:
	rm -f *.o main




		
