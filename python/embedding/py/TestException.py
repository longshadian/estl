
import http.client
import json
import random
import time
import datetime
import configparser
import os
import threading

import UpdateSelf

def TestIni():
    print(os.path.abspath(__file__))
    try:
        with open("./a.txt", "rb") as f:
            print(f.read())
    except (BaseException) as e:
        print("exception:xxx {}".format(e))


def Main():
    while True:
        #TestIni()
        print(UpdateSelf.CheckUpdateSelf())
        time.sleep(2)


if __name__ == '__main__':
    Main()
