#!/bin/bash
wget https://www.dropbox.com/sh/y1qr9jkp74m9lm2/AADbBssbTv7sbL4sHBaPjS7ua?dl=1 -O lib/w2v_model
unzip download/w2v_model -d lib/w2v_model
python final_predict.py $1 $2 $3
