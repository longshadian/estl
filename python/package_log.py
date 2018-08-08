<<<<<<< HEAD
#!/usr/bin/python3 -B

import os,sys
import zipfile
import time,datetime
from datetime import date

SAVE_TIME_CYCLE = (2 * 7)
SLEEP_PRE_TIME  = (1 * 60 * 60)


def zip_file(zip_path, file_path, filename):
    """
    :param zip_path: path+zipname.zip
    :param file_path: path+logfile.log
    :param filename: logfile.log
    :return:none
    """
    print("zip file:", filename)
    zip = zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED)
    for name in zip.namelist():
        if name == filename:
            return
    zip.write(file_path, filename)
    zip.close();


def zip_game(path):
    """
    :param path:game log dir
    :return:none
    """
    todaydate = datetime.datetime.now()

    try:
        filelist = os.listdir(path)
    except FileNotFoundError:
        print("path not found:", path)
        return

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            continue
        if ".log" not in filename and ".txt" not in filename:
            continue
        index = filename.rfind(".log")
        if index == -1:
            continue
        filestr = filename[0:index]
        filedate = datetime.datetime.strptime(filestr, "%Y%m%d")
        date_diff = todaydate - filedate
        if date_diff.days >= SAVE_TIME_CYCLE:
            zip_path = path + filestr[0:-2] + ".zip"
            file_path = path + filename
            zip_file(zip_path, file_path, filename)
            os.remove(filepath)

def zip_tomcat(path):
    """
    :param path:tomcat log dir
    :return:none
    """
    todaydate = datetime.datetime.now()

    try:
        filelist = os.listdir(path)
    except FileNotFoundError:
        print("path not found:", path)
        return

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            continue
        if ".log" not in filename and ".txt" not in filename:
            continue
        first_dot_index = filename.find(".")
        secnd_dot_index = filename.rfind(".")
        if first_dot_index == secnd_dot_index:
            continue
        file_pre = filename[0:first_dot_index]
        file_time = filename[first_dot_index + 1:secnd_dot_index]
        filedate = datetime.datetime.strptime(file_time, "%Y-%m-%d")
        date_diff = todaydate - filedate
        if date_diff.days >= SAVE_TIME_CYCLE:
            zip_path = path + file_pre + "." + file_time[0:-3] + ".zip"
            file_path = path + filename
            zip_file(zip_path, file_path, filename)
            os.remove(filepath)

#脚本运行后每天晚上3点钟将指定日志目录下指定周期之外的日志打包到当月日志压缩文件中并删除本地日志
def main():
    print("process begin ....")
    zip_game("/home/cgy/work/log/ddz/log/")
    zip_game("/home/cgy/work/log/ddzpersistance/log/")
    zip_tomcat("/home/cgy/work/log/tomcat/log/")
    print("process end ....")

if __name__ == '__main__':
    main()
=======
#!/usr/bin/python3 -B

import os,sys
import zipfile
import time,datetime
from datetime import date

SAVE_TIME_CYCLE = (2 * 7)
SLEEP_PRE_TIME  = (1 * 60 * 60)


def zip_file(zip_path, file_path, filename):
    """
    :param zip_path: path+zipname.zip
    :param file_path: path+logfile.log
    :param filename: logfile.log
    :return:none
    """
    print("zip file:", filename)
    zip = zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED)
    for name in zip.namelist():
        if name == filename:
            return
    zip.write(file_path, filename)
    zip.close();


def zip_game(path):
    """
    :param path:game log dir
    :return:none
    """
    todaydate = datetime.datetime.now()

    try:
        filelist = os.listdir(path)
    except FileNotFoundError:
        print("path not found:", path)
        return

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            continue
        if ".log" not in filename and ".txt" not in filename:
            continue
        index = filename.rfind(".log")
        if index == -1:
            continue
        filestr = filename[0:index]
        filedate = datetime.datetime.strptime(filestr, "%Y%m%d")
        date_diff = todaydate - filedate
        if date_diff.days >= SAVE_TIME_CYCLE:
            zip_path = path + filestr[0:-2] + ".zip"
            file_path = path + filename
            zip_file(zip_path, file_path, filename)
            os.remove(filepath)

def zip_tomcat(path):
    """
    :param path:tomcat log dir
    :return:none
    """
    todaydate = datetime.datetime.now()

    try:
        filelist = os.listdir(path)
    except FileNotFoundError:
        print("path not found:", path)
        return

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            continue
        if ".log" not in filename and ".txt" not in filename:
            continue
        first_dot_index = filename.find(".")
        secnd_dot_index = filename.rfind(".")
        if first_dot_index == secnd_dot_index:
            continue
        file_pre = filename[0:first_dot_index]
        file_time = filename[first_dot_index + 1:secnd_dot_index]
        filedate = datetime.datetime.strptime(file_time, "%Y-%m-%d")
        date_diff = todaydate - filedate
        if date_diff.days >= SAVE_TIME_CYCLE:
            zip_path = path + file_pre + "." + file_time[0:-3] + ".zip"
            file_path = path + filename
            zip_file(zip_path, file_path, filename)
            os.remove(filepath)

#脚本运行后每天晚上3点钟将指定日志目录下指定周期之外的日志打包到当月日志压缩文件中并删除本地日志
def main():
    print("process begin ....")
    zip_game("/home/cgy/work/log/ddz/log/")
    zip_game("/home/cgy/work/log/ddzpersistance/log/")
    zip_tomcat("/home/cgy/work/log/tomcat/log/")
    print("process end ....")

if __name__ == '__main__':
    main()
>>>>>>> 555bdac81e27e7244b3d29153f9a3ab67b08c357
