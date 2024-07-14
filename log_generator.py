# This generates log files

import os
import glob
import random
import string
from datetime import datetime

# 'WARNING', 'ERROR', 'CRITICAL'


def get_current_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def format_log_entry(time, level, message):
    log_entry = f"{time} - {level}: {message}"
    return log_entry


def generate_ok_log_entry():
    log_level = ['DEBUG', 'INFO']
    random_log_level = random.choice(log_level)
    message_length = random.randint(20, 50)  # random length for the message
    message = ''.join(random.choices(string.ascii_letters, k=message_length))  # generate random message

    return format_log_entry(get_current_time(), random_log_level, message)


def generate_error_log_entry():
    message_length = random.randint(20, 50)  # random length for the message
    error_msg = ''.join(random.choices(string.ascii_letters, k=message_length))

    return format_log_entry(get_current_time(), 'ERROR', "An error occurred: " + error_msg)


def make_ok_logs():
    for file_idx in range(10):
        with open(f"./logs/ok-{file_idx}.log", "w") as file:
            for _ in range(500):
                file.write(generate_ok_log_entry() + "\n")


def make_error_logs():
    for file_idx in range(10):
        with open(f"./logs/err-{file_idx}.log", "w") as file:
            for line_idx in range(500):
                if line_idx % 20 == 0:
                    file.write(generate_error_log_entry() + "\n")
                else:
                    file.write(generate_ok_log_entry() + "\n")


def make_clean_log_dir():
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    files = glob.glob('./logs/*')
    for f in files:
        os.remove(f)


def make_new_log_files():
    make_clean_log_dir()
    make_ok_logs()
    make_error_logs()


if __name__ == '__main__':
    make_new_log_files()
