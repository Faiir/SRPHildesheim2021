import os


def check_for_statusmanager(log_dir, file_name):
    return os.path.exists(os.path.join(log_dir, file_name))