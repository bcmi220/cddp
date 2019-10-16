#!/bin/bash

for i in 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    let i4=i-4
    let i3=i-3
    let i2=i-2
    let i1=i-1
    python ./examples/run_dp.py --data_dir=./data/NLPCC/Unlabeled/ --test_file=PC-Unlabeled.conll --bert_model=./outputs/PC_un${i2}_finetune/,./outputs/PC_un${i3}_finetune/,./outputs/PC_un${i4}_finetune/ --max_seq_length=320 --do_ensemble_predict --test_batch_size=128 --label_vocab=./data/NLPCC/Train/labels.vocab --test_output=./data/NLPCC/Train/PC_un_${i}.pred
    python scripts/merge_tri_training.py ./data/NLPCC/Train/BC-Train.conll ./data/NLPCC/Train/PC_un_${i}.pred ./data/NLPCC/Train/PC-Train-split-un${i}.conll
    python ./examples/run_dp.py --data_dir=./data/NLPCC/Train/ --train_file=PC-Train-split-un${i}.conll --val_file=PC-Dev-split.conll --bert_model=./outputs/PC_un${i1}_finetune/ --output_dir=outputs/PC_un${i}_finetune/ --max_seq_length=320 --do_train --do_eval --train_batch_size=128 --eval_batch_size=128 --learning_rate=1e-5 --num_train_epochs=3 --label_vocab=./data/NLPCC/Train/labels.vocab --seed=${i} --has_confidence
    python ./examples/run_dp.py --data_dir=./data/NLPCC/Train/ --test_file=PC-Dev-split.conll --bert_model=./outputs/PC_un${i}_finetune/ --max_seq_length=320 --do_predict --test_batch_size=128 --label_vocab=./data/NLPCC/Train/labels.vocab --test_output=./result/PC_dev_${i}.out --output_dir=./outputs/
    echo "${i}:"
    perl scripts/eval.pl -g ./data/NLPCC/Train/PC-Dev-split.conll -s ./result/PC_dev_${i}.out -q
    echo -e "\n\n"
done