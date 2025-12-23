#!/bin/bash
method=pcsodv2
eval_folder=/root/autodl-tmp/PCSOD/pcsod/'one-key evaluation'

# generate saliency maps
cd code
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
