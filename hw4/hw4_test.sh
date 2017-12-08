#!/bin/bash
wget https://www.dropbox.com/s/nb1z9gzzr00uuxd/Dictonary_all_2.bin?dl=1 -O MyDictionary.bin
python hw4_test.py $1 $2
