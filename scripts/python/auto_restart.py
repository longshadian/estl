###!/usr/bin/python3 -B

import subprocess
import os
import sys
import time
import datetime


BIN_NAME = "/home/cgy/work/xxx/cldldwss-1.4.jar --env=local7100"
FULL_CMD = "nohup java -Xmx10g -Xms10g -XX:+PrintGCDetails -Xloggc:gc.log -jar {name} > /dev/null 2>&1 &"

def full_name():
    return FULL_CMD.format(name = BIN_NAME)

def bin_name():
    return BIN_NAME

def get_tm():
    return datetime.datetime.today().isoformat(sep=' ')

def get_pid():
    pid = subprocess.check_output("ps x|grep '{name}' | grep -v 'grep' | awk '{{print $1}}'".format(name=bin_name()), shell=True, universal_newlines=True)
    if pid:
        pid = pid[:-1]
    return pid

def start():
    pid = get_pid()
    tm = get_tm()
    if pid:
        print("{tm} {name} is running already! pid:{pid}".format(tm=tm, name=bin_name(), pid=pid))
        return
    subprocess.Popen([full_name()], shell=True, universal_newlines=True)
    time.sleep(2)
    pid = get_pid()
    if pid:
        print("{tm} {name} start success! pid:{pid}".format(tm=tm, name=bin_name(), pid=pid))
    else:
        print("{tm} {name} start failed!".format(tm=tm, name=bin_name()))

def stop():
    tm = get_tm()
    pid = get_pid()
    if not pid:
        print("{tm} {name} isn't running!".format(tm=tm, name=bin_name()))
        return

    subprocess.Popen(["kill", pid], universal_newlines=True)
    print("{tm} start kill {name} pid:{pid}".format(tm=tm, name=bin_name(), pid=pid))
    while True:
        time.sleep(1)
        pid = get_pid()
        if not pid:
            print("{tm} stop {name} success!".format(tm=tm, name=bin_name()))
            return
        print("{tm} waiting {name} stop pid:{pid}".format(tm=tm, name=bin_name(), pid=pid))

def main(args):
    usage = "usage:\n    auto_restart.py [start] [stop]"
    if len(args) != 2:
        print(usage)
        return    
    cmd = args[1]
    if cmd == "start":
        start()
    elif cmd == "stop":
        stop()
    else:
        print("unknown cmd: {}".format(cmd))
        print(usage)
        

if __name__ == '__main__':
    main(sys.argv)
    #print(s)
