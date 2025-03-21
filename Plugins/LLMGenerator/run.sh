#!/bin/bash
plugin_dir=$1  # First positional argument
other_args=$2  # Second positional argument

source $plugin_dir/venv/bin/activate
python3 $plugin_dir/export.py
