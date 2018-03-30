#!/usr/bin/python3 -B

import shutil
import os
import datetime
import subprocess


def main():
    base_src_path = "/home/cgy/work/ddzgame/src2"
    base_dst_path = "./src2"
    copy_dir = ["ddz", "common", "ddz_persistance", "tools"]
    ignore_pattern = shutil.ignore_patterns("*.o", "*.cc", "*.a", "ddzserver", "ddzpst")

    if os.path.exists(base_dst_path):
        shutil.rmtree(base_dst_path)
    os.mkdir(base_dst_path)

    for entry in copy_dir:
        src = os.path.join(base_src_path, entry)
        dst = os.path.join(base_dst_path, entry)
        shutil.copytree(src, dst, ignore=ignore_pattern)

    t = datetime.datetime.now()
    tar_name = t.strftime("ddzgame2-%m-%d-%H-%M.tar.gz")
    cmd = "tar -cf {0} ./src2/*".format(tar_name)
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
