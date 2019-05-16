
INC =  	-I/usr/local/mysql/include

LIB	= -lmysqlpp -L/usr/local/mysql/lib -lmysqlclient_r
		
CC      = g++
CFLAGS  = -g -Wall -Wshadow -Wunreachable-code  -Wextra -Wconversion -Wl,--no-as-needed -Werror -std=c++11 

.SUFFIXES:  .o .cpp .cc

.PHONY:all clean

DIR_SRC	= .
SRC_CPP = $(wildcard $(DIR_SRC)/*.cpp)
INC += -I$(DIR_SRC)

OBJS = $(SRC_CPP:%.cpp=%.o)

OUTPUT = a.out

all:$(OBJS)
	$(CC) $(CFLAGS) $(INC) $^ -o $(OUTPUT) $(LIB)

$(OUTPUT): $(OBJS)
	$(CC) $(CFLAGS) $(INC) $^ -o $@ $(LIB)

.cpp.o:
	$(CC) $(CFLAGS) $(INC) -c $< -o $@ 

.cc.o:
	$(CC) $(CFLAGS) $(INC) -c $< -o $@ 
		
clean: 
	echo $(OBJS)
	rm -f $(OUTPUT)
	rm -f $(OBJS)
	find ./ -regex ".*\.o$$" | xargs echo
	find ./ -regex ".*\.o$$" | xargs rm -f
