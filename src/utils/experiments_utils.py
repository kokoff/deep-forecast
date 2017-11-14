import os
from distutils import dir_util

EXPERIMENTS_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'experiments'))


def expr_sub_dir(*sub_path):
    path = os.path.join(EXPERIMENTS_DIR, os.path.sep.join(sub_path))

    if not os.path.exists(path):
        dir_util.mkpath(path)

    return path


def expr_file_name(*args):
    if len(args) < 2:
        raise TypeError('Not enough arguments.')

    file_name = '_'.join(args[:-1]) + '.' + args[-1]

    return file_name


def main():
    pass


if __name__ == '__main__':
    main()
