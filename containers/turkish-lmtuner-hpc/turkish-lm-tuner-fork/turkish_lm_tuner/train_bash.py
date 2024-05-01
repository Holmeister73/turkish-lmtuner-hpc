# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:09:58 2024

@author: USER
"""

import argparse, sys

parser=argparse.ArgumentParser()

parser.add_argument("--bar", help="Do the bar option")
parser.add_argument("--foo", help="Foo the program")

args=parser.parse_args()

print(f"Dict format: {vars(args)}")
