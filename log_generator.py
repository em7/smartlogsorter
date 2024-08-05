# This generates log files

import os
import glob
import random
import string
from datetime import datetime

# 'WARNING', 'ERROR', 'CRITICAL'

# key is the error message, value is context dict - pre and post
logs_dictionary = {
    'TypeError: \'NoneType\' object is not iterable': {
        'pre': ['Traceback (most recent call last):',
                '    File "bla-bla-bla.py", line 49, in <module>',
                '        for x in mylist:', ],
        'post': []
    },
    'RecursionError: maximum recursion depth exceeded in comparison': {
        'pre': [
            "Traceback (most recent call last):",
            '  File "stack_overflow_example.py", line 4, in <module>',
            "    recursive_function()",
            '  File "stack_overflow_example.py", line 2, in recursive_function',
            "    return recursive_function()",
            '  File "stack_overflow_example.py", line 2, in recursive_function',
            "    return recursive_function()",
            '  File "stack_overflow_example.py", line 2, in recursive_function',
            "    return recursive_function()",
            '  File "stack_overflow_example.py", line 2, in recursive_function',
            "    return recursive_function()",
            '  File "stack_overflow_example.py", line 2, in recursive_function',
            "    return recursive_function()",
            '  File "stack_overflow_example.py", line 2, in recursive_function',
            "    return recursive_function()",
            '  File "stack_overflow_example.py", line 2, in recursive_function',
            "    return recursive_function()",
            '  ...',
            '  File "stack_overflow_example.py", line 2, in recursive_function',
            "    return recursive_function()"
        ],
        'post': []
    },
    'KeyError: \'some_key\'': {
        'pre': ["Traceback (most recent call last):",
                '  File "key_error_example.py", line 6, in <module>',
                "    key_error_example()",
                '  File "key_error_example.py", line 4, in key_error_example',
                '    return sample_dict["d"]'
                ],
        'post': []
    },
    'TypeError: \'int\' object does not support item assignment': {
        'pre': ["Traceback (most recent call last):",
                '  File "type_error_example.py", line 5, in <module>',
                "    type_error_example()",
                '  File "type_error_example.py", line 3, in type_error_example',
                '    x = "This is a string"'],
        'post': []
    },
    'TypeError: \'int\' object is not subscriptable': {
        'pre': ["Traceback (most recent call last):",
                '  File "subscript_error_example.py", line 5, in <module>',
                "    subscript_error_example()",
                '  File "subscript_error_example.py", line 3, in subscript_error_example',
                "    return num[0]"],
        'post': []
    },
}


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
    error_msg = random.choice(list(logs_dictionary.keys()))
    context_pre = logs_dictionary[error_msg]['pre']
    context_post = logs_dictionary[error_msg]['post']

    return context_pre\
        + [format_log_entry(get_current_time(), 'ERROR', "An error occurred: " + error_msg)]\
        + context_post


def make_clean_log_dir():
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    files = glob.glob('./logs/*')
    for f in files:
        os.remove(f)


def generate_log_with_error():
    log_messages = []
    for _ in range(random.randint(100, 150)):
        log_messages.append(generate_ok_log_entry())

    for line in generate_error_log_entry():
        log_messages.append(line)

    for _ in range(random.randint(100, 150)):
        log_messages.append(generate_ok_log_entry())

    return log_messages


def output_log_file(filename, log_messages):
    with open(filename, 'w') as file:
        for message in log_messages:
            file.write(message + '\n')


def make_new_log_files():
    make_clean_log_dir()

    for i in range(50):
        log_messages = generate_log_with_error()
        filename = f'./logs/log_{i}.txt'
        output_log_file(filename, log_messages)



if __name__ == '__main__':
    make_new_log_files()
