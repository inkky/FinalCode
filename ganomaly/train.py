"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly
from lib.data_preprocess_KDD import load_data_kdd
from lib.data_preprocess_arr import load_data_arr
from lib.data_preprocess_ucr import load_data_ucr

##
# def main():
""" Training
"""

##
# ARGUMENTS
opt = Options().parse()
print(opt.anomaly_class)
##
# LOAD DATA
# dataloader = load_data(opt)
# dataloader = load_data_kdd(opt)
# dataloader = load_data_arr(opt)
dataloader = load_data_ucr(opt)

##
# LOAD MODEL
model = Ganomaly(opt, dataloader)

##
# TRAIN MODEL
model.train()


# if __name__ == '__main__':
#     main()
