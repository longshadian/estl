
CXXFLAG:=-Wall -g -std=c++11


OBJ_CREATE:=main.cpp \
	SharedMemoryAPI.cpp
	
OBJ_DEL:=delmain.cpp \
	SharedMemoryAPI.cpp
	
OBJ_OPEN:=openmain.cpp \
	SharedMemoryAPI.cpp
	
INC_CREATE:=-I./

all:clean create open del

TARGET_CREATE:=create
TARGET_DEL:=del
TARGET_OPEN:=open

LINK_LIB:=-lrt -lpthread


create:
	g++ $(CXXFLAG) $(OBJ_CREATE) -o $(TARGET_CREATE) $(LINK_LIB)

open:
	g++ $(CXXFLAG) $(OBJ_OPEN) -o $(TARGET_OPEN) $(LINK_LIB)
	
del:
	g++ $(CXXFLAG) $(OBJ_DEL) -o $(TARGET_DEL) $(LINK_LIB)
	
clean:
	rm -f ./$(TARGET_CREATE) ./$(TARGET_DEL) ./$(TARGET_OPEN)
