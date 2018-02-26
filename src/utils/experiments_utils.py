import os

_BACKUP_DIR = os.path.abspath(os.path.join('../../.experiments_backup'))
EXPERIMENTS_DIR = os.path.abspath(os.path.join('../../experiments'))

if not os.path.exists(EXPERIMENTS_DIR):
    os.mkdir(EXPERIMENTS_DIR)
