cd /share03/xzli/btask
CUDA_VISIBLE_DEVICES=`/share03/securityL2/PBStools/idle-gpus.pl -n 8` /share03/xzli/miniconda3/envs/pytorch/bin/python ./examples/run_nlpcc_dp.py --bert_model ./outputs/NLPCC_All_PC2/ --max_seq_length 300 --do_predict --test_batch_size 512 --label_vocab ./data/NLPCC/Train/labels.vocab --test_file ./data/NLPCC/Unlabeled/PC-Unlabeled.all.conll --test_output ./result/t4-pc2.pred