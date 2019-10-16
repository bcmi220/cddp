#!/bin/bash

for i in 16 17 18 19 20
do
    let i4=i-4
    let i3=i-3
    let i2=i-2
    let i1=i-1
    python ./examples/run_dp.py --data_dir=./data/NLPCC/Unlabeled/ --test_file=ZX-Unlabeled-merge.conll --bert_model=./outputs/ZX_un${i2}_finetune/,./outputs/ZX_un${i3}_finetune/,./outputs/ZX_un${i4}_finetune/ --max_seq_length=320 --do_ensemble_predict --test_batch_size=32 --label_vocab=./data/NLPCC/Train/labels.vocab --test_output=./data/NLPCC/Train/ZX_un_${i}.pred
    python scripts/merge_tri_training.py ./data/NLPCC/Train/ZX-Train-split-merge.conll ./data/NLPCC/Train/ZX_un_${i}.pred ./data/NLPCC/Train/ZX-Train-split-merge-un${i}.conll
    python ./examples/run_dp.py --data_dir=./data/NLPCC/Train/ --train_file=ZX-Train-split-merge-un${i}.conll --val_file=ZX-Dev-split.conll --bert_model=./outputs/ZX_un${i1}_finetune/ --output_dir=outputs/ZX_un${i}_finetune/ --max_seq_length=320 --do_train --do_eval --train_batch_size=128 --eval_batch_size=128 --learning_rate=1e-5 --num_train_epochs=1 --label_vocab=./data/NLPCC/Train/labels.vocab --seed=${i} --has_confidence
    python ./examples/run_dp.py --data_dir=./data/NLPCC/Train/ --test_file=ZX-Dev-split.conll --bert_model=./outputs/ZX_un${i}_finetune/ --max_seq_length=320 --do_predict --test_batch_size=128 --label_vocab=./data/NLPCC/Train/labels.vocab --test_output=./result/ZX_dev_${i}.out --output_dir=./outputs/
    echo "${i}:"
    perl scripts/eval.pl -g ./data/NLPCC/Train/ZX-Dev-split.conll -s ./result/ZX_dev_${i}.out -q
    echo -e "\n\n"
done