
INC = -I/home/cgy/work/lib_src/boost_1_59_0/

LIB	= -L/home/cgy/work/lib_src/boost_1_59_0/stage/lib -lboost_system -lpthread
		
CC      = g++
##CFLAGS  = -g -Wall -Wshadow -Wunreachable-code  -Wextra -Wconversion -Wl,--no-as-needed -Werror -std=c++11 
CFLAGS  = -g -Wall -Wunreachable-code -Werror -std=c++11 
LDFLAGS = -Wl,--no-as-needed -Wl,-rpath /home/cgy/work/lib_src/boost_1_59_0/stage/lib

.SUFFIXES:  .o .cpp .cc

.PHONY:all clean

DIR_SRC	= .
SRC_CPP = $(wildcard $(DIR_SRC)/*.cpp)
INC += -I$(DIR_SRC)

OBJS = $(SRC_CPP:%.cpp=%.o)

OUTPUT = a.out

all:$(OBJS)
	$(CC) $(CFLAGS) $(INC) $^  $(LIB) $(LDFLAGS) -o $(OUTPUT)

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
