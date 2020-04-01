
import random
import CppModule

def Initialize():
    print("script Initialize")
    
def Cleanup():
    print("script Cleanup")

def HttpRequest():
    print("script Cleanup")
    
def Add(a,b):
    print("aaaa ", random.randint(1,11111))
    return a + b + 100000
	
def Length(a):
	return len(a)

name="xxxxxxx"

def TestCppNew():
    p = CppModule.Pet("Cat")
    print("pet1: ", p.GetName())
    print("Subtract: ", CppModule.Subtract(1,2))

def TestCppObject():    
    p = CppModule.GlobalPet()
    print("pet2: ", p.GetName())
    print("random", random.randint(1, 100))
    p.SetName(str(random.randint(1, 100)))

#TestCppObject()


