
import os
import sys

import LogSig
import IPLoM
import SLCT
import LKE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')

input_dir    = 'logs/HDFS/' # The input directory of log file
output_dir   = 'LogSig_result/' # The output directory of parsing results
log_file     = 'HDFS_2k.log' # The input log file name
log_format   = '<Date> <Time> <Pid> <Level> <Component>: <Content>' # HDFS log format
regex        = []  # Regular expression list for optional preprocessing (default: [])
group_number = 14 # The number of message groups to partition

parser = LogSig.LogParser(input_dir, output_dir, group_number, log_format, rex=regex)
parser.parse(log_file)

support    = 10  # The minimum support threshold
parser = SLCT.LogParser(log_format=log_format, indir=input_dir, outdir=output_dir, 
                        support=support, rex=regex)
parser.parse(log_file)