
import random
import Cpp


def OnCall(hdl):
    S = Cpp.GetStack(hdl)
    #print("OnCall size: ", S.Size())
    if S.Size() == 0:
        S.Push("abc")
    elif S.Size() == 1:
        S.Push(111.12233)
    elif S.Size() == 2:
        S.Push(['a', 'b'])
    elif S.Size() == 3:
        S.Set(0, random.randint(0,100000))
        S.Push("xxxxxx")
    else:
        S.Pop()
    StackToString(S)
    val = S.Get(111)
    if val is None:
        print("val is index: ", val)
    return 0

def StackToString(S):
    buf = []
    for i in range(0, S.Size()):
        buf.append(S.Get(i))
    print(buf)
    

