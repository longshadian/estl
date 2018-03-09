
INC =   -I./ -I/usr/local/include/zookeeper

LIB     =  -lzookeeper_mt

CC      = g++
CFLAGS  = -g -Wall -std=c++14

.SUFFIXES:  .o .cpp .cc

.PHONY:clean


all:client server


client:client.cpp
	$(CC) $(CFLAGS) $(INC) $^ -o $@ $(LIB)

server:server.cpp
	$(CC) $(CFLAGS) $(INC) $^ -o $@ $(LIB)


#.cpp.o:
#        $(CC) $(CFLAGS) $(INC) -c $< -o $@

.PHONY:clean

clean:
	rm client server

