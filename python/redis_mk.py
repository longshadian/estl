#!/usr/bin/python3 -B

import os
import sys
import subprocess

REDIS_CLI = './redis-cli'
REDIS_HOST = '127.0.0.1'
REDIS_PORT = 6379

REDIS_SERVER = "./redis-server"
REDIS_CONF = './redis.conf'

def get_client_cmd():
    return "{cli} -h {host} -p {port}".format(cli=REDIS_CLI, host=REDIS_HOST, port=REDIS_PORT)

def get_redis_client():
    cmd = get_client_cmd()
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=True, universal_newlines=True)

def ping():
    redis = get_redis_client()
    out, err = redis.communicate("PING")
    if err:
        print(err)
        return
    for row in out.split():
        print(row)
    redis.wait()

def keys():
    redis = get_redis_client()
    out, err = redis.communicate("KEYS *")
    if err:
        print(err)
        return
    for row in out.split():
        print(row)
    redis.wait()

def start():
    cmd = "{server} {conf}".format(server=REDIS_SERVER, conf=REDIS_CONF)
    subprocess.call(cmd, shell=True, universal_newlines=True)

def shutdown():
    redis = get_redis_client()
    out, err = redis.communicate("shutdown")
    if err:
        print(err)
        return
    redis.wait()

def main(args):
    cmd_list = {
        "start": start,
        "shutdown": shutdown,
        "ping": ping,
        "keys": keys
    }
    usage = '''
Usage: ./redis_mk.py [options]
    start               启动redis server
    shutdown            redis-cli: SHUTDOWN
    ping                redis-cli: PING
    keys                redis-cli: KEYS *
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

if __name__ == '__main__':
    main(sys.argv)
