import os
import shutil


def copyFiles(old_path, new_path):
    for ipath in os.listdir(old_path):
        fulldir = os.path.join(old_path, ipath)
        if os.path.isfile(fulldir):
            shutil.copy(fulldir, new_path)
        if os.path.isdir(fulldir):
            copyFiles(fulldir, new_path)
