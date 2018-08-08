#!/usr/bin/python3 -B

import shutil
import os
import datetime
import subprocess
import zipfile

###打包源代码
def packageSrc():
    base_src_path = ".."
    base_dst_path = "src"
    copy_dir = ["proto", "share", "src", "utils", "src_charge"]
    ignore_pattern = shutil.ignore_patterns("*.o", "*.cc", "*.a", "zjhserver", "zjh_charge")

    if os.path.exists(base_dst_path):
        shutil.rmtree(base_dst_path)
    os.mkdir(base_dst_path)

    for entry in copy_dir:
        src = os.path.join(base_src_path, entry)
        dst = os.path.join(base_dst_path, entry)
        print("复制文件:from[{}] to[{}]".format(src, dst))
        shutil.copytree(src, dst, ignore=ignore_pattern)

    '''
    t = datetime.datetime.now()
    tar_name = t.strftime("src-%m-%d-%H-%M.tar.gz")
    cmd = "tar -cf {0} {1}/*".format(tar_name, base_dst_path)
    subprocess.call(cmd, shell=True)
    '''
    t = datetime.datetime.now()
    tar_name = t.strftime("src-%m-%d-%H-%M")
    print("压缩文件:{}.zip".format(tar_name))
    shutil.make_archive(tar_name, 'zip', base_dir=base_dst_path)
    shutil.rmtree(base_dst_path)
    print("打包完成:{}.zip".format(tar_name))

def main():
    packageSrc()

if __name__ == '__main__':
    main()
