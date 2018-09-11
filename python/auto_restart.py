#!/usr/bin/python36 -B

import subprocess
import os
import sys
import time
import datetime


NAME = "'./gamebox_v2-1.3.jar --env=prod'"
CMD = "nohup java -XX:+HeapDumpOnOutOfMemoryError -jar ./gamebox_v2-1.3.jar --env=prod > /dev/null 2>&1 &"

def main(args):
    start()

def get_tm():
    return datetime.datetime.today().isoformat(sep=' ')

def get_pid():
    pid = subprocess.check_output("ps x|grep {name} | grep -v 'grep' | awk '{{print $1}}'".format(name=NAME), shell=True, universal_newlines=True)
    if pid:
        pid = pid[:-1]
    return pid

def start():
    pid = get_pid()
    tm = get_tm()
    if pid:
        print("{tm} {name} is running already! pid:{pid}".format(tm=tm, name=NAME, pid=pid))
        return
    subprocess.Popen([CMD], shell=True, universal_newlines=True)
    time.sleep(2)
    pid = get_pid()
    if pid:
        print("{tm} {name} start success! pid:{pid}".format(tm=tm, name=NAME, pid=pid))
    else:
        print("{tm} {name} start failed!".format(tm=tm, name=NAME))

def stop():
    tm = get_tm()
    pid = get_pid()
    if not pid:
        print("{tm} {name} isn't running!".format(tm=tm, name=NAME))
        return

    subprocess.Popen(["kill", pid], universal_newlines=True)
    print("{tm} start kill {name} pid:{pid}".format(tm=tm, name=NAME, pid=pid))
    while True:
        time.sleep(1)
        pid = get_pid()
        if not pid:
            print("{tm} stop {name} success!".format(tm=tm, name=NAME))
            return
        print("{tm} waiting {name} stop pid:{pid}".format(tm=tm, name=NAME, pid=pid))

if __name__ == '__main__':
    main(sys.argv)
    #print(s)
