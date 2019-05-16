
INC =	-I/usr/local/mysql/include		\
		-I./ -I../../

LIB	= -L../ -lmysqlcpp -L/usr/local/mysql/lib -lmysqlclient
		
CC      = g++
CFLAGS  = -g -Wall -Werror -Wextra -std=c++14
#  -Wshadow -Wunreachable-code -Wextra -Wconversion
LDFLAGS = -Wl,--no-as-needed,-rpath=/usr/local/mysql/lib -lpthread

.SUFFIXES:  .o .cpp .cc

.PHONY:all install clean

SRC_DIR	= .
SRC_CPP = $(wildcard $(SRC_DIR)/*.cpp)
SRC_CC = $(wildcard $(SRC_DIR)/*.cc)
INC += -I$(SRC_DIR)

OUTPUT = test_xx2 test_all test_threads test_safe_string test_convert test_big_table

all:$(OUTPUT)

test_xx2:TestXX2.cpp
	$(CC) $(CFLAGS) $(INC) $^ $(LIB) $(LDFLAGS) -o $@

test_all:TestAll.cpp
	$(CC) $(CFLAGS) $(INC) $^ $(LIB) $(LDFLAGS) -o $@

test_threads:TestThreads.cpp
	$(CC) $(CFLAGS) $(INC) $^ $(LIB) $(LDFLAGS) -o $@

test_safe_string:TestSafeString.cpp
	$(CC) $(CFLAGS) $(INC) $^ $(LIB) $(LDFLAGS) -o $@

test_convert:TestConvert.cpp
	$(CC) $(CFLAGS) $(INC) $^ $(LIB) $(LDFLAGS) -o $@
    
test_big_table:TestBigTable.cpp
	$(CC) $(CFLAGS) $(INC) $^ $(LIB) $(LDFLAGS) -o $@
    

clean: 
	@rm -f $(OUTPUT)
	@find ./ -regex ".*\.o$$" | xargs rm -f
	
