#!/usr/bin/python3

import shutil
import datetime
import os
import re
import subprocess
import sys


def ls_dir_files(src):
    dirs = []
    files = []
    for entry in os.listdir(src):
        abs_path = os.path.join(src, entry)
        if os.path.isdir(abs_path):
            if entry != "." or entry != "..":
                dirs.append(entry)
        elif os.path.isfile(abs_path):
            files.append(entry)
    return dirs, files


def copy_dir_by_re(src, dst, reg_pattern=""):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.mkdir(dst)
    dirs, files = ls_dir_files(src)
    for f in files:
        if re.search(reg_pattern, f):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    for d in dirs:
        copy_dir_by_re(os.path.join(src, d), os.path.join(dst, d), reg_pattern)


def builds_ddz_ex():
    base_src_path = "/home/cgy/work/ddzgame/src_tags/1.4.1"
    base_dst_path = "./src2"
    copy_dir = ["ddz", "common", "ddz_persistance", "tools"]
    reg_pattern = "\w+\.h$|\w+\.cpp$|^Makefile$"

    if os.path.exists(base_dst_path):
        shutil.rmtree(base_dst_path)
    os.mkdir(base_dst_path)

    for entry in copy_dir:
        copy_dir_by_re(os.path.join(base_src_path, entry), os.path.join(base_dst_path, entry), reg_pattern)

    t = datetime.datetime.now()
    tar_name = t.strftime("ddzgame2-%m-%d-%H-%M.tar.gz")
    cmd = "tar -cf {0} ./src2/*".format(tar_name)
    subprocess.call(cmd, shell=True)


def osPopen():
    lines = os.popen('ps x|grep a.x | grep -v "grep"').readlines()
    if len(lines) == 0:
        print("not run")
    else:
        for f in lines:
            print("len:{0} {1}".format(len(f), f))


def subProcess():
    #subprocess.call(["tar", "-cvf", "x.tar.gz", "./a.py"])
    #subprocess.call("rm -rf ./temp/*", shell=True)
    #subprocess.check_call(["cp", "-r", "./temp_ex", "./temp"])
    #subprocess.call("cd ./temp_ex", shell=True)
    #subprocess.call(["ls", "-l"])
    #s = subprocess.check_output("ps x|grep ./bin/ddzserver2 |grep -v 'grep' | awk '{print $1}'", shell=True, universal_newlines=True)

    #child = subprocess.Popen(["./b.out"], stdout=subprocess.PIPE, universal_newlines=True)
    child = subprocess.Popen(["./b.out"], stdout=subprocess.PIPE, universal_newlines=True)
    child.wait()
    for s in child.stdout.readlines():
        print("python {0}".format(s))

    #child.communicate("xxxfff")
    print("child exit")



def main():
    #builds_ddz_ex()
    subProcess()

if __name__ == '__main__':
    #print(sys.argv[0])
    #print(sys.argv[1])

    main()


def test_ls_dir_files(src):
    dirs, files = ls_dir_files(src)
    for entry in dirs:
        print("{0:5} {1}".format("dir", entry))

    for entry in files:
        print("{0:5} {1}".format("file", entry))