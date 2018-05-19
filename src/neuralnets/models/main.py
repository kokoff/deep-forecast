import subprocess
from itertools import product
from src.utils.data_utils import VARIABLES
import psutil
import os
import sys

one_one_in = [([i], [i]) for i in VARIABLES]
many_one = [(VARIABLES, [i]) for i in VARIABLES]
many_many = [(VARIABLES, VARIABLES)]


def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print output.strip()
    rc = process.poll()
    return rc


def kill_others():
    for proc in psutil.process_iter():
        pinfo = proc.as_dict(attrs=['pid', 'name'])
        procname = str(pinfo['name'])
        procpid = str(pinfo['pid'])
        if "python" in procname and procpid != str(os.getpid()):
            print("Stopped Python Process ", proc)
            proc.kill()


def mlp_experiments():
    countries = ['EA', 'US']
    vars = one_one_in + many_one + many_many

    for i, j in product(countries, vars):
        args = ['python', '-m', 'scoop', 'run_models.py', '-m', 'mlp', '-c', i, '--in', ' '.join(j[0]), '--out',
                ' '.join(j[1])]
        run_command(args)
        kill_others()


def lstm_experiments():
    countries = ['EA', 'US']
    vars = one_one_in

    for i, j in product(countries, vars):
        print i, j
        args = ['python', '-m', 'scoop', 'run_models.py', '-m', 'lstm', '-c', i, '--in', ' '.join(j[0]), '--out',
                ' '.join(j[1])]
        run_command(args)
        kill_others()


if __name__ == '__main__':
    # lstm_experiments()
    mlp_experiments()
