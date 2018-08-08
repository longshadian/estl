<<<<<<< HEAD
#!/usr/bin/python3 -B

import os, sys

#将指定目录下所有文件拷贝到指定目标目录
def copyfile(src_dir, dest_dir):
    """
    :param src_dir:copy src dir
    :param dest_dir:copy to dest dir
    :return: none
    """
    filelist = os.listdir(src_dir)
    for filename in filelist:
        src_file = os.path.join(src_dir, filename)
        dest_file= os.path.join(dest_dir, filename)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        if os.path.isdir(src_file):
            copyfile(src_file, dest_file)
        else:
            if not os.path.exists(dest_file):
                print("copy file:", src_file, " to ", dest_dir)
                open(dest_file, "wb").write(open(src_file, "rb").read())
            else:
                print("warning dest file:", dest_file, " is exists")

if __name__ == '__main__':
    copyfile("/home/cgy/work/log", "/home/cgy/work/log2")

=======
#!/usr/bin/python3 -B

import os, sys

#将指定目录下所有文件拷贝到指定目标目录
def copyfile(src_dir, dest_dir):
    """
    :param src_dir:copy src dir
    :param dest_dir:copy to dest dir
    :return: none
    """
    filelist = os.listdir(src_dir)
    for filename in filelist:
        src_file = os.path.join(src_dir, filename)
        dest_file= os.path.join(dest_dir, filename)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        if os.path.isdir(src_file):
            copyfile(src_file, dest_file)
        else:
            if not os.path.exists(dest_file):
                print("copy file:", src_file, " to ", dest_dir)
                open(dest_file, "wb").write(open(src_file, "rb").read())
            else:
                print("warning dest file:", dest_file, " is exists")

if __name__ == '__main__':
    copyfile("/home/cgy/work/log", "/home/cgy/work/log2")

>>>>>>> 555bdac81e27e7244b3d29153f9a3ab67b08c357
