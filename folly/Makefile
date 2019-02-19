
INC = 	-I./ -I/home/chen/project/folly-master/
		
CC      = g++
CFLAGS  = -g -Wall -std=c++11

LIB =  -lglog -L//usr/local/lib/ -lfolly -levent -ldl -liberty -lboost_context -lrt -lpthread

.SUFFIXES:
.SUFFIXES:  .o .cpp .cc

.PHONY:all install clean

SRC_CPP = $(wildcard ./*.cpp)

OBJS = $(SRC_CPP:%.cpp=%.o)

OUTPUT = test
#all:$(OBJS)
#	ar -rc ./libgame.a $(OBJS) 

all:$(OBJS)
	$(CC) $(CFLAGS) $^ -o $(OUTPUT) $(LIB)
	
$(OUTPUT): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIB)

.cpp.o:
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

.cc.o:
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

clean: 
	rm -f $(OUTPUT)
	find ./ -regex ".*\.o$$" | xargs echo
	find ./ -regex ".*\.o$$" | xargs rm -f
	
