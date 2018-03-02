#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHONPATH=${PYTHONPATH}: $(realpath$DIR/../../)
echo $PYTHONPATH
