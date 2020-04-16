
INC = 	-I../

LIB	= 	-lboost_filesystem -lboost_system -ldl -lstdc++fs

		
CC      = g++
CFLAGS  = -g -fPIC -Wall -Wextra -Wconversion -Wshadow -Werror -std=c++17 -DNDEBUG

.SUFFIXES:  .o .cpp .cc

.PHONY:all test clean

DIR_SRC	= .
SRC_CPP = $(wildcard $(DIR_SRC)/*.cpp)
INC += -I$(DIR_SRC)

OBJS = $(SRC_CPP:%.cpp=%.o)

OUTPUT = test_blacklist

all:$(OUTPUT)

$(OUTPUT): $(OBJS)
	$(CC) -g $^ -o $@ $(LIB)

.cpp.o:
	@echo [  $(shell expr $(shell echo $(OBJS)|sed 's#$@.*$$#$@#'|wc -w) \* 100 / $(words $(OBJS)))%] $(CC) $(CFLAGS) -c $< -o $@ 
	@$(CC) $(CFLAGS) $(INC) -c $< -o $@ 

test:
	#cp $(OUTPUT) ~/work/test
	@echo nothing to do
		
clean: 
	@echo $(OBJS)
	@rm -f $(OUTPUT)
	@rm -f $(OBJS)
	@find ./ -regex ".*\.o$$" | xargs rm -f
	
