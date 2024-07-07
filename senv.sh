#!/bin/bash

SCRIPT_NAME=$(basename $0)
echo "Usage: source $SCRIPT_NAME"

source venv/bin/activate

export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib64/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"