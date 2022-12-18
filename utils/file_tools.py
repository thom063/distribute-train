import io
import os
import pickle
import shutil
from typing import List

from utils.filelock import FileLock


def save_buff(objs: List[object], file_path: str):
    file_path_lock = file_path + ".lock"
    with FileLock(file_path_lock):
        with open(file_path, "wb") as f:
            for obj in objs:
                pickle.dump(obj, f)

def load_file(path: str) -> List[object]:
    file_path_lock = path + ".lock"
    with FileLock(file_path_lock):
        out_list = []
        f = open(path, 'rb')
        while 1:
            try:
                out_list.append(pickle.load(f))
            except EOFError:
                break
    return out_list


def set_dir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)