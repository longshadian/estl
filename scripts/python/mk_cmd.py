#!/usr/bin/python3 -B

import subprocess
import os
import sys
import time


SRC = "./bin/ddzserver2"


def main(args):
    cmd_list = {
        "start": start,
        "stop": stop,
        "restart": restart,
        "--sr": server_cmd,
        "--cr": server_cmd,
    }
    usage = '''
Usage: ./mk.py [options]
        start               启动服务        如果服务已经运行，会提示无法启动"
        stop                停止服务        如果该服务没有运行，不会有提示"
        restart             先停止服务,后启动服务"
        --sr                保存redis缓存中所有需要保存的玩家数据,需要先stop服务"
        --cr                删除redis缓存中所有已经保存的玩家数据,需要先stop服务"
'''

    if len(args) == 1:
        print("args less!")
        print(usage)
        return
    elif len(args) > 2:
        print("args too much!")
        print(usage)
        return

    if args[1] not in cmd_list:
        print("unknown options")
        print(usage)
        return

    cmd_list[args[1]]()


def get_pid():
    pid = subprocess.check_output("ps x|grep {src} | grep -v 'grep' | awk '{{print $1}}'".format(src=SRC), shell=True, universal_newlines=True)
    if pid:
        pid = pid[:-1]
    return pid


def start():
    pid = get_pid()
    if pid:
        print("{0} is running already! pid:{1}".format(SRC, pid))
        return
    subprocess.Popen([SRC], universal_newlines=True)
    time.sleep(2)
    pid = get_pid()
    if pid:
        print("{0} start success! pid:{1}".format(SRC, pid))
    else:
        print("{0} start failed!".format(SRC))


def restart():
    pid = get_pid()
    if pid:
        print("{0} is running! pid:{1}".format(SRC, pid))
        stop()
    print("start {0}".format(SRC))
    start()
    pass


def stop():
    pid = get_pid()
    if not pid:
        print("{0} isn't running!".format(SRC))
        return

    subprocess.Popen(["kill", pid], universal_newlines=True)
    print("start kill {0} pid:{1}".format(SRC, pid))
    while True:
        time.sleep(1)
        pid = get_pid()
        if not pid:
            print("stop {0} success!".format(SRC))
            return
        print("waiting {0} stop pid:{1}".format(SRC, pid))

def server_cmd():
    pid = get_pid()
    if pid:
        print("{0} is running! pid:{1}".format(SRC, pid))
        return
    cmd = SRC
    for i in range(0, len(sys.argv)):
        if i == 0:
            continue
        cmd += " "
        cmd += sys.argv[i]
    #print(cmd)
    subprocess.Popen([cmd], shell=True, universal_newlines=True)

if __name__ == '__main__':
    main(sys.argv)
    #print(s)
