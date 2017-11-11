"""
Plots MaxF1 score.
-------------------------------------------------
The MIT License (MIT)
Copyright (c) 2017 Marvin Teichmann
More details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
import sys

import matplotlib.pyplot as plt

runfolder = '/home/pmmn11/Experiments/KittiSeg/RUNS/mini_batch'
currun = '8k_b2_KittiSeg_mini-batch_2017_10_17_20.35'
anafile = 'output.log'

output_folder = '/home/pmmn11/Experiments/KittiSeg/Results'
name = currun.split('_')[0]

outname = os.path.join(output_folder, name)

filename = os.path.join(runfolder, currun, anafile)
eval_iters = 500
max_iters = 8000


def read_values(prop, typ):
    regex_string = "%s\s+\(%s\)\s+:\s+(\d+\.\d+)" % (prop, typ)
    regex = re.compile(regex_string)

    value_list = [regex.search(line).group(1) for line in open(filename)
                  if regex.search(line) is not None]

    float_list = [float(value) for value in value_list]
    print(len(float_list[1::2]))
    return float_list


label_list = range(eval_iters, max_iters+1, eval_iters)
#label_list = (1, 1);
print(len(label_list))
plt.figure(figsize=(8, 5))

plt.rcParams.update({'font.size': 14})



plt.plot(label_list, read_values('MaxF1', 'raw')[1::2],
         label='MaxF1 (Raw)', color='blue', marker=".", linestyle=' ')

plt.plot(label_list, read_values('MaxF1', 'smooth')[1::2],
         label='MaxF1 (Smoothed)', color='blue')

plt.plot(label_list, read_values('Average Precision', 'raw')[1::2],
         label='Average Precision (Raw)', color='red', marker=".",
         linestyle=' ')
plt.plot(label_list, read_values('Average Precision', 'smooth')[1::2],
         label='Average Precision (Smoothed)', color='red')


plt.yticks(np.arange(90, 100, 0.5))

plt.xlabel('Iteration')

plt.ylabel('Validation Score [%]')
plt.legend(loc=0)

#plt.savefig(outname + ".pdf")
plt.savefig(outname + ".png")

plt.show()
