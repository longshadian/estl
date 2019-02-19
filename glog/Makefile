
INC =    

LIB	= 	-lglog -lpthread 
		
CC      = g++
CFLAGS  = -g -Wall -Wshadow  -Wunreachable-code  -Wextra -Wconversion -Wl,--no-as-needed -Werror -std=c++11 

.SUFFIXES:  .o .cpp .cc

.PHONY:all install clean

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
	@echo [  $(shell expr $(shell echo $(OBJS)|sed 's#$@.*$$#$@#'|wc -w) \* 100 / $(words $(OBJS)))%] $(CC) $(CFLAGS) -c $< -o $@ 
	@$(CC) $(CFLAGS) $(INC) -c $< -o $@ 

clean: 
	@echo $(OBJS)
	@rm -f $(OUTPUT)
	@rm -f $(OBJS)
	@find ./ -regex ".*\.o$$" | xargs rm -f
	
