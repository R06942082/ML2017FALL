#!/bin/bash
mkdir download 
wget https://www.dropbox.com/sh/y1qr9jkp74m9lm2/AADbBssbTv7sbL4sHBaPjS7ua?dl=1 -O download/w2v_model
mkdir lib/w2v_model
unzip download/w2v_model -d lib/w2v_model
python final_predict.py $1 $2 $3
