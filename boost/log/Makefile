
LIB	= 	-lboost_date_time 	\
		-lboost_log 		\
		-lboost_log_setup	\
		-lboost_system 		\
		-lpthread 			\
		-lboost_thread
		
CC      = 	g++
CFLAGS  = 	-g -Wall -Werror -std=c++11  -DBOOST_LOG_DYN_LINK
LDFLAGS = 	-Wl,--no-as-needed -Wl
#,-rpath=$(DEPENDS)/boost_1_59_0/stage/lib


.SUFFIXES:  .o .cpp

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

clean: 
	echo $(OBJS)
	rm -f $(OUTPUT)
	rm -f $(OBJS)
	find ./ -regex ".*\.o$$" | xargs echo
	find ./ -regex ".*\.o$$" | xargs rm -f
