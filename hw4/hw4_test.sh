#!/bin/bash
wget https://www.dropbox.com/s/yqjoqddge0014a8/Dictonary_all_2.bin?dl=1 -O MyDictionary.bin
python hw4_test.py $1 $2
