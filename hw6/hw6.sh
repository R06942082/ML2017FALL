#!/bin/bash
wget https://www.dropbox.com/s/hbvtkmotsw1ne9d/epoch-738_acc-0.010.hdf5?dl=1 -O epoch-738_acc-0.010.hdf5
python hw6.py $1 $2 $3
