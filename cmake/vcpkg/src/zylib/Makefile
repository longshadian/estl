INC = 	-I./
		
LIB	= 
		
CC      = g++
CFLAGS  = -g -Wall -Wextra -Werror -std=c++11 -DBOOST_LOG_DYN_LINK

.SUFFIXES:  .o .cpp .cc

.PHONY:clean

DIR_SRC= .
SRC_CPP = $(wildcard $(DIR_SRC)/*.cpp)

OBJS = $(SRC_CPP:%.cpp=%.o)

OUTPUT = ./libzylib.a

$(OUTPUT): $(OBJS)
	ar -rc $(OUTPUT) $(OBJS) 
	
.cpp.o:
	@echo [  $(shell expr $(shell echo $(OBJS)|sed 's#$@.*$$#$@#'|wc -w) \* 100 / $(words $(OBJS)))%] $(CC) -c $< -o $@ 
	@$(CC) $(CFLAGS) $(INC) -c $< -o $@
	
clean: 
	rm -f $(OUTPUT)
	find ./ -regex ".*\.o$$" | xargs echo
	find ./ -regex ".*\.o$$" | xargs rm -f
