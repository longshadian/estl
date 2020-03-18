
INC =  	-I/usr/local/modsecurity/include	\
		-I/home/cgy/Desktop/cinatra/include \
		-I./

LIB	= 	-L/usr/local/modsecurity/lib	\
		-lmodsecurity 					\
		-lstdc++fs \
		-lboost_system \
		-luuid \
		-lpthread 

		
CC      = g++
CFLAGS  = -g -Wl,-rpath=/usr/local/modsecurity/lib -Werror -std=c++17 -DNDEBUG

.SUFFIXES:  .o .cpp .cc

.PHONY:all install clean

DIR_SRC	= .
SRC_CPP = $(wildcard $(DIR_SRC)/*.cpp)
INC += -I$(DIR_SRC)

############## common dir ##############
#SRC_CPP += $(wildcard ../common/proto/*.cpp)

############## subdir ##############
#DIR_WORLD=./world
#SRC_CPP += $(wildcard $(DIR_WORLD)/*.cpp)
#INC += -I$(DIR_WORLD)

###################################################

OBJS = $(SRC_CPP:%.cpp=%.o)

OUTPUT = test_modsecurity.bin

all:$(OBJS)
	$(CC) $(CFLAGS) $(INC) $^ -o $(OUTPUT) $(LIB)

$(OUTPUT): $(OBJS)
	$(CC) $(CFLAGS) $(INC) $^ -o $@ $(LIB)

.cpp.o:
	@echo [  $(shell expr $(shell echo $(OBJS)|sed 's#$@.*$$#$@#'|wc -w) \* 100 / $(words $(OBJS)))%] $(CC) $(CFLAGS) -c $< -o $@ 
	@$(CC) $(CFLAGS) $(INC) -c $< -o $@ 

install:
	@echo "error" 
		
clean: 
	@echo $(OBJS)
	@rm -f $(OUTPUT)
	@rm -f $(OBJS)
	@find ./ -regex ".*\.o$$" | xargs rm -f
	
