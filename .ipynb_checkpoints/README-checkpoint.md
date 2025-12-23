# test using author weight
[usage] 
1. generate sal maps: 
run 'python test.py --log_dir pcsod --data_root /root/autodl-tmp/PCSOD/data/' (under './code')
2. copy sal maps to eval folder:
run 'method=pcsod && cp log/${method}/visual/* ../'one-key evaluation'/pred/${method}/PCSOD/' (under './code')
3. metric eval: 
run 'python main.py' (under './one-key evaluation')
[result] the performance is the same as reported in the paper.
PCSOD dataset with PointSal method get 0.0684 mae, 0.7718 adap-fmeasure, 0.7721 mean-fmeasure, 0.8043 max-fmeasure,  0.8767 adap-Emeasure, 0.8533 mean-Emeasure, 0.8796 max-Emeasure, 0.6582 IoU..

# test by reproducing training
[usage] 
train: 
python train.py --log_dir pcsodv2 --data_root /root/autodl-tmp/PCSOD/data/
test: 
1. python test.py --log_dir 2022-11-28_23-35 --data_root /root/autodl-tmp/PCSOD/data/ (under './code')
2. method=2022-11-28_23-35 && cp log/${method}/visual/* ../'one-key evaluation'/pred/${method}/PCSOD/ (under './code')
3. python main.py (under './one-key evaluation')
or 
simply run: 'bash test.sh' (under './') 
[result] performance is the same as reported in the paper. (some are bettter, e.g., IoU, meanEmeasure)
PCSOD dataset with 2022-11-28_23-35 method get 0.0677 mae, 0.7758 adap-fmeasure, 0.7775 mean-fmeasure, 0.8267 max-fmeasure,  0.8803 adap-Emeasure, 0.8556 mean-Emeasure, 0.9011 max-Emeasure, 0.6646 IoU..

# test other methods performance by author saliency maps.
[result] all compared methods performance are the same as reported. (except that PointCNN is much lower)

# versions
[pcsod] test succ using author weights; test succ by reproduce training.
[pcsod-v1] using a separate folder to develop code, transfer result to pcsod for testing.
changes: modify test script 'test.sh'.
[pcsod-v2] add some comments to show feature map size.
