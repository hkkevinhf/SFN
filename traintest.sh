#!/bin/bash
# set method name and eval path
method=$(basename $(pwd)) 
eval_folder=/root/autodl-tmp/PCSOD/pcsod/'one-key evaluation'

# train the model
cd code
python train.py --log_dir ${method} --data_root /root/autodl-tmp/PCSOD/data/ --batch_size 32 --epoch 3000

# generate saliency maps
python test.py --log_dir ${method} --data_root /root/autodl-tmp/PCSOD/data/

# generate evaluation scores
eval_path="${eval_folder}"/pred/${method}/PCSOD
if [ -d "${eval_path}" ]; then
    rm -f "${eval_path}"/*
else
    mkdir -p "${eval_path}"
fi
cp log/${method}/visual/*  "${eval_path}"/
cd "${eval_folder}"
python main.py --method ${method}
