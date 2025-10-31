import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="") 
parser.add_argument("--target", type=str, default="cuDNN")
args = parser.parse_args()
path = args.path
target = args.target
last_type = ""
last_N = ""
next_type = ""
next_N = ""

with open(path, "r") as f:
    for line in f:
        # Average time: 0.043 ms
        pattern = r"Average time: (\d+\.\d+) ms"
        match = re.search(pattern, line)
        # Type= 6float2, N = 64
        pattern2 = r"Type= (\d+\w+), N = (\d+)"
        match2 = re.search(pattern2, line)
        if match2:
            next_type = match2.group(1)
            next_N = match2.group(2)
        if next_type != last_type or next_N != last_N:
            print(f"Type {next_type}, N = {next_N}")
            last_type = next_type
            last_N = next_N
        if match:
            print(match.group(1))
