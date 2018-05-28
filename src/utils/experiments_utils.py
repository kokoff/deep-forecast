import os
import json

config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config.json')
with open(config_file, 'r') as f:
    config = json.load(f)

EXPERIMENTS_DIR = config['EXPERIMENTS_DIR']

if not os.path.isabs(EXPERIMENTS_DIR):
    EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(config_file), EXPERIMENTS_DIR))

if not os.path.exists(EXPERIMENTS_DIR):
    os.mkdir(EXPERIMENTS_DIR)

print 'EXPERIMENTS_DIR = ', EXPERIMENTS_DIR
