#!/usr/bin/python3 -B

import shutil
import os
import datetime
import subprocess

##打包源代码
def packageSrc():
    base_src_path = "./.."
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
    cmd = "tar -cf {0} {1}/*".format(tar_name, base_dst_path)
    subprocess.call(cmd, shell=True)
    shutil.rmtree(base_dst_path)

##打包数据库，不包含用户数据
def packageSql():
    ##测试服
    #db_ip = "120.76.118.91"
    #db_port = 3817
    #db_passwd = "Mydb@Jy11#07"

    ##内网
    db_ip = "127.0.0.1"
    db_port = 3306
    db_passwd = "123456"
    db_user = "root"
    db_name = "ddzgame"
    #表结构和数据
    db_table = ("tapk",
                "tbbs",
                "tcard_policy",
                "tchat",
                "tcoin_tree",
                "tconf",
                "texchange",
                "tguid",
                "tiptable",
                "tiptable_robot",
                "titem",
                "tlevel_experience",
                "tlottery_item",
                "tmission",
                "tnickname",
                "tnn_room",
                "tprop_conf",
                "trank_robot",
                "trobot_chat",
                "troom",
                "tshared_conf",
                "tshop",
                "tsign_reward",
                "ttitle",
                "tvip",
                "tword_filter")

    #表结构
    db_table_empty = (
        "tuser_0",
        "tuser_1",
        "tuser_2",
        "tuser_3",
        "tuser_4",
        "tuser_5",
        "tuser_6",
        "tuser_7",
        "tuser_8",
        "tuser_9",
        "tuser_basic_data_0",
        "tuser_basic_data_1",
        "tuser_basic_data_2",
        "tuser_basic_data_3",
        "tuser_basic_data_4",
        "tuser_basic_data_5",
        "tuser_basic_data_6",
        "tuser_basic_data_7",
        "tuser_basic_data_8",
        "tuser_basic_data_9",
        "tuser_exchange_0",
        "tuser_exchange_1",
        "tuser_exchange_2",
        "tuser_exchange_3",
        "tuser_exchange_4",
        "tuser_exchange_5",
        "tuser_exchange_6",
        "tuser_exchange_7",
        "tuser_exchange_8",
        "tuser_exchange_9",
        "tuser_game_data_0",
        "tuser_game_data_1",
        "tuser_game_data_2",
        "tuser_game_data_3",
        "tuser_game_data_4",
        "tuser_game_data_5",
        "tuser_game_data_6",
        "tuser_game_data_7",
        "tuser_game_data_8",
        "tuser_game_data_9",
        "tuser_item_0",
        "tuser_item_1",
        "tuser_item_2",
        "tuser_item_3",
        "tuser_item_4",
        "tuser_item_5",
        "tuser_item_6",
        "tuser_item_7",
        "tuser_item_8",
        "tuser_item_9",
        "tuser_mission_0",
        "tuser_mission_1",
        "tuser_mission_2",
        "tuser_mission_3",
        "tuser_mission_4",
        "tuser_mission_5",
        "tuser_mission_6",
        "tuser_mission_7",
        "tuser_mission_8",
        "tuser_mission_9",
        "tuser_comment")

    base_dst_path = "./sql"
    if os.path.exists(base_dst_path):
        shutil.rmtree(base_dst_path)
    os.mkdir(base_dst_path)

    opt = " --skip-comments --skip-extended-insert"
    for t in db_table:
        cmd = "mysqldump -u {db_user} -p{db_passwd} -P {db_port} {opt} --host={db_ip} {db_name} {table} > {file}/{table}.sql"\
            .format(
            opt = opt,
            db_user=db_user,
            db_passwd=db_passwd,
            db_port=db_port,
            db_ip=db_ip,
            db_name=db_name,
            table=t,
            file=base_dst_path
        )
        #print(cmd)
        subprocess.call(cmd, shell=True)

    for t in db_table_empty:
        cmd = "mysqldump -u {db_user} -p{db_passwd} -P {db_port} {opt} --host={db_ip} {db_name} -d {table} > {file}/{table}.sql" \
            .format(
            opt=opt,
            db_user=db_user,
            db_passwd=db_passwd,
            db_port=db_port,
            db_ip=db_ip,
            db_name=db_name,
            table=t,
            file=base_dst_path
        )
        # print(cmd)
        subprocess.call(cmd, shell=True)

    t = datetime.datetime.now()
    tar_name = t.strftime("sql-%m-%d-%H-%M.tar.gz")
    cmd = "tar -cf {0} {1}*".format(tar_name, base_dst_path)
    subprocess.call(cmd, shell=True)
    shutil.rmtree(base_dst_path)

def main():
    packageSql()

if __name__ == '__main__':
    main()
