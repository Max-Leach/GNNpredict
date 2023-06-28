#! /bin/bash

bash 1_setup.sh
bash 2_match.sh
python cross-check.py
