
INC =   -I./

LIB     =  -lzookeeper_mt

CC      = g++
CFLAGS  = -g -Wall -std=c++14 -DTHREADED

.SUFFIXES:  .o .cpp .cc

.PHONY:clean


all:client server client2


client:client.cpp
	$(CC) $(CFLAGS) $(INC) $^ -o $@ $(LIB)

server:server.cpp
	$(CC) $(CFLAGS) $(INC) $^ -o $@ $(LIB)

client2:client2.cpp
	$(CC) $(CFLAGS) $(INC) $^ -o $@ $(LIB)


#.cpp.o:
#        $(CC) $(CFLAGS) $(INC) -c $< -o $@

.PHONY:clean

clean:
	rm ./client ./server ./client2

