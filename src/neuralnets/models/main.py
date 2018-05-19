import subprocess
from itertools import product
from src.utils.data_utils import VARIABLES
import psutil
import os
import sys
import argparse

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


def mlp_experiments(diff=True):
    countries = ['EA', 'US']
    vars = one_one_in + many_one + many_many

    for i, j in product(countries, vars):
        args = ['python', '-m', 'scoop', 'run_models.py', '-m', 'mlp', '-c', i, '--in'] + j[0] + ['--out'] + j[1]

        if diff:
            args.append('-d')

        run_command(args)
        kill_others()


def lstm_experiments(diff=True):
    countries = ['EA', 'US']
    vars = one_one_in

    for i, j in product(countries, vars):
        print i, j
        args = ['python', '-m', 'scoop', 'run_models.py', '-m', 'lstm', '-c', i, '--in'] + j[0] + ['--out'] + j[1]

        if diff:
            args.append('-d')

        run_command(args)
        kill_others()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['mlp', 'lstm', 'all'], required=True)
    parser.add_argument('-d', '--diff', choices=['yes', 'no', 'both'], required=True)
    args = parser.parse_args()
    args = vars(args)
    print args

    if args['diff'] == 'both' or args['diff'] == 'yes':
        if args['model'] == 'all' or args['model'] == 'lstm':
            lstm_experiments(diff=True)
        if args['model'] == 'all' or args['model'] == 'mlp':
            mlp_experiments(diff=True)

    if args['diff'] == 'both' or args['diff'] == 'no':
        if args['model'] == 'all' or args['model'] == 'lstm':
            lstm_experiments(diff=False)
        if args['model'] == 'all' or args['model'] == 'mlp':
            mlp_experiments(diff=False)
