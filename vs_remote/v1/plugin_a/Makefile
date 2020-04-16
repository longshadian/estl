
INC = 	-I../

LIB	=  -lrocksdb -lmaxminddb -lstdc++fs

		
CC      = g++
CFLAGS  = -g -fPIC -Wall -Wextra -Wconversion -Wshadow -Werror -std=c++17 -DNDEBUG

.SUFFIXES:  .o .cpp .cc

.PHONY:all install clean

DIR_SRC	= .
SRC_CPP = $(wildcard $(DIR_SRC)/*.cpp)
INC += -I$(DIR_SRC)

OBJS = $(SRC_CPP:%.cpp=%.o)

OUTPUT = blacklist.so

all:$(OUTPUT)

$(OUTPUT): $(OBJS)
	$(CC) -g -shared $(CFLAGS) $(INC) $^ -o $@ $(LIB)

.cpp.o:
	@echo [  $(shell expr $(shell echo $(OBJS)|sed 's#$@.*$$#$@#'|wc -w) \* 100 / $(words $(OBJS)))%] $(CC) $(CFLAGS) -c $< -o $@ 
	@$(CC) $(CFLAGS) $(INC) -c $< -o $@ 

install:
	cp $(OUTPUT) ~/work/test
		
clean: 
	@echo $(OBJS)
	@rm -f $(OUTPUT)
	@rm -f $(OBJS)
	@find ./ -regex ".*\.o$$" | xargs rm -f
	
