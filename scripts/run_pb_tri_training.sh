#!/bin/bash

# python ./examples/run_dp.py --data_dir=./data/NLPCC/Unlabeled/ --test_file=PB-Unlabeled-merge.conll --bert_model=./outputs/PB_un_finetune/ --max_seq_length=320 --do_predict --test_batch_size=128 --label_vocab=./data/NLPCC/Train/labels.vocab --test_output=./data/NLPCC/Train/PB_un_3.pred --output_dir=./outputs/
# python scripts/merge_tri_training.py ./data/NLPCC/Train/PB-Train-split-merge.conll ./data/NLPCC/Train/PB_un_3.pred ./data/NLPCC/Train/PB-Train-split-merge-un3.conll
# python ./examples/run_dp.py --data_dir=./data/NLPCC/Train/ --train_file=PB-Train-split-merge-un3.conll --val_file=PB-Dev-split.conll --bert_model=./outputs/PB_un2_finetune/ --output_dir=outputs/PB_un3_finetune/ --max_seq_length=320 --do_train --do_eval --train_batch_size=128 --eval_batch_size=128 --learning_rate=1e-5 --num_train_epochs=1 --label_vocab=./data/NLPCC/Train/labels.vocab --seed=3 --has_confidence
# python ./examples/run_dp.py --data_dir=./data/NLPCC/Train/ --test_file=PB-Dev-split.conll --bert_model=./outputs/PB_un3_finetune/ --max_seq_length=320 --do_predict --test_batch_size=128 --label_vocab=./data/NLPCC/Train/labels.vocab --test_output=./result/PB_dev_3.out --output_dir=./outputs/
# echo "3:"
# perl scripts/eval.pl -g ./data/NLPCC/Train/PB-Dev-split.conll -s ./result/PB_dev_3.out -q
# echo -e "\n\n"

for i in 3 4 5
do
    let i2=i-2
    let i1=i-1
    python ./examples/run_dp.py --data_dir=./data/NLPCC/Unlabeled/ --test_file=PB-Unlabeled-merge.conll --bert_model=./outputs/PB_un${i2}_finetune/ --max_seq_length=320 --do_predict --test_batch_size=128 --label_vocab=./data/NLPCC/Train/labels.vocab --test_output=./data/NLPCC/Train/PB_n_${i}.pred
    python scripts/merge_tri_training.py ./data/NLPCC/Train/PB-Train-split-merge.conll ./data/NLPCC/Train/PB_n_${i}.pred ./data/NLPCC/Train/PB-Train-split-merge-n${i}.conll
    python ./examples/run_dp.py --data_dir=./data/NLPCC/Train/ --train_file=PB-Train-split-merge-n${i}.conll --val_file=PB-Dev-split.conll --bert_model=./outputs/PB_n${i1}/ --output_dir=outputs/PB_n${i}/ --max_seq_length=320 --do_train --do_eval --train_batch_size=128 --eval_batch_size=128 --learning_rate=1e-5 --num_train_epochs=1 --label_vocab=./data/NLPCC/Train/labels.vocab --seed=${i} --has_confidence
    python ./examples/run_dp.py --data_dir=./data/NLPCC/Train/ --test_file=PB-Dev-split.conll --bert_model=./outputs/PB_n${i}/ --max_seq_length=320 --do_predict --test_batch_size=128 --label_vocab=./data/NLPCC/Train/labels.vocab --test_output=./result/PB_n_${i}.out --output_dir=./outputs/
    echo "${i}:"
    perl scripts/eval.pl -g ./data/NLPCC/Train/PB-Dev-split.conll -s ./result/PB_dev_${i}.out -q
    echo -e "\n\n"
done