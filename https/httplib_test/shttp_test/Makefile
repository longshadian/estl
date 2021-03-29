

MOD_PATH=$(shell pwd)


INC = -I./ \
      -I$(PRJ_PATH)/external/boost/include 

LIB	= -L/home/bolan/libs/lib -lboost_filesystem  -lboost_system \
      -ldl -lpthread
		
CC      = g++
CFLAGS  = -g -Wall -Wextra -Wconversion -std=c++11
CFLAGS_CC = -g -Wall -Wextra -Wconversion -std=c++11
LDFLAGS = -Wl,-rpath=$(PRJ_PATH)/libs/boost/lib

.SUFFIXES:  .o .cpp .cc

.PHONY:all install clean

SRC_DIR	= .
SRC_CPP = $(wildcard $(SRC_DIR)/*.cpp)
SRC_CC = $(wildcard $(SRC_DIR)/*.cc)

###################################################

OBJS = $(SRC_CPP:%.cpp=%.o)
OBJS += $(SRC_CC:%.cc=%.o)

OUTPUT = PAS_controlHttp

all:$(OBJS)
	$(CC) $(CFLAGS) $(INC) $^ $(LIB) $(LDFLAGS) -o $(OUTPUT)
	mkdir -p $(PRJ_PATH)/../bin
	mv -vf $(OUTPUT)  $(PRJ_PATH)/../bin

$(OUTPUT): $(OBJS)
	$(CC) $(CFLAGS) $(INC) $^ -o $@ $(LIB)

.cpp.o:
	@echo [  $(shell expr $(shell echo $(OBJS)|sed 's#$@.*$$#$@#'|wc -w) \* 100 / $(words $(OBJS)))%] $(CC) -c $< -o $@ 
	@$(CC) $(CFLAGS) $(INC) -c $< -o $@ 
	
	
.cc.o:
	@echo [  $(shell expr $(shell echo $(OBJS)|sed 's#$@.*$$#$@#'|wc -w) \* 100 / $(words $(OBJS)))%] $(CC) -c $< -o $@ 
	@$(CC) $(CFLAGS_CC) $(INC) -c $< -o $@ 
    
	
#install:
#	cp -f $(OUTPUT) /home/cgy/szmj_server/szmj/bin/
    

clean:
	@echo $(OBJS)
	@rm -f $(OUTPUT)
	@rm -f $(OBJS)
	@find ./ -regex ".*\.o$$" | xargs rm -f
	
