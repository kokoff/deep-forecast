import subprocess
from itertools import product
import sys


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


if __name__ == '__main__':
    countries = ['EA', 'US']
    vars = ['one_one', 'many_one', 'many_many']
    lags = ['4', '8']

    for i, j, k in product(countries, vars, lags):
        args = ['python', '-m', 'scoop', 'run_models.py', i, j, k]
        run_command(args)
