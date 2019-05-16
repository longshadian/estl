
MYSQL_INCLUDE=/usr/local/mysql/include

INC =  -I$(MYSQL_INCLUDE) -I../
		
LIB	= 
		
CC      = g++
CFLAGS  = -g -Wall -Wextra -Werror -std=c++11

.SUFFIXES:  .o .cpp .cc

.PHONY:clean

DIR_SRC= .
SRC_CPP = $(wildcard $(DIR_SRC)/*.cpp)
SRC_CPP += $(wildcard ./detail/*.cpp)

OBJS = $(SRC_CPP:%.cpp=%.o)

OUTPUT = ./libmysqlcpp.a

$(OUTPUT): $(OBJS)
	ar -rc $(OUTPUT) $(OBJS) 
	
.cpp.o:
	@echo [  $(shell expr $(shell echo $(OBJS)|sed 's#$@.*$$#$@#'|wc -w) \* 100 / $(words $(OBJS)))%] $(CC) -c $< -o $@ 
	@$(CC) $(CFLAGS) $(INC) -c $< -o $@
	
clean: 
	rm -f $(OUTPUT)
	find ./ -regex ".*\.o$$" | xargs echo
	find ./ -regex ".*\.o$$" | xargs rm -f
