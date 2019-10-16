from __future__ import absolute_import

import argparse
import csv
import logging
import os
import random
import sys
from io import open
import re
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

sys.path.append('.')

from bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from bert.modeling import BertForDependencyParsing, BertConfig
from bert.optimization import BertAdam, WarmupLinearSchedule
from bert.tokenization import BertTokenizer

logger = logging.getLogger(__name__)

class ConllExample(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 sent_id,
                 sentence,
                 postags=None,
                 heads = None,
                 labels = None,
                 confidence=None): # label confidence we need this to control the loss
        self.sent_id = sent_id
        self.sentence = sentence
        self.postags = postags
        self.heads = heads
        self.labels = labels
        self.confidence = confidence

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "sent_id: {}".format(self.sent_id),
            "sentence: {}".format(self.sentence),
        ]
        if self.postags is not None:
            l.append("postags: {}".format(self.postags))
        if self.heads is not None:
            l.append("heads: {}".format(self.heads))
        if self.labels is not None:
            l.append("labels: {}".format(self.labels))
        if self.confidence is not None:
            l.append("confidence: {}".format(self.confidence))

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 input_ids,
                 input_mask,
                 segment_ids,
                 heads,
                 labels,
                 confidence,
                 seq_len,
                 word_index,
                 token_starts,
                 token_ends,
                 example
    ):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.heads = heads
        self.labels = labels
        self.confidence = confidence
        self.seq_len = seq_len
        self.word_index = word_index
        self.token_starts = token_starts
        self.token_ends = token_ends
        self.example = example

def read_conll_examples(input_file, is_training, has_confidence):
    data = []
    block = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) == 0:
                if len(block) > 0:
                    data.append(block)
                    block = []
            else:
                block.append(line.strip().split('\t'))
        if len(block) > 0:
            data.append(block)

    examples = [
        ConllExample(
            sent_id = i,
            sentence = [line[1] for line in sent],
            postags = [line[3] for line in sent],
            heads = [int(line[6]) for line in sent] if is_training else None,
            labels = [line[7] for line in sent] if is_training else None,
            confidence= [float(line[9]) for line in sent] if is_training and has_confidence else None,
        ) for i,sent in enumerate(data) # we skip the line with the column names
    ]

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, label_vocab2idx,
                                 is_training, has_confidence):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    skip_num = 0
    example_index = 0
    for example in examples:
        # because our sentence have been tokenized, so we just do subword tokenize on it

        #sentence:          I   am  a   student
        #tokenized:         I   am  a   stu@@   dent
        #index:             0   1   2   3
        #tokenized index:   0   1   2   3       3
        #head:              2   0   4   2
        #tokenized head:    
        tokens, word_index, token_starts, token_ends = tokenizer.sub_tokenize(example.sentence)

        if len(tokens) > (max_seq_length - 2):
            skip_num += 1
            continue

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        word_index = [0] + [idx+1 for idx in word_index] + [len(example.sentence)+1] # token -> word
        token_starts = [0] + [idx+1 for idx in token_starts] + [len(tokens)+1] # word -> token starts
        token_ends = [0] + [idx+1 for idx in token_ends] + [len(tokens)+1] # word -> token starts
        # reverse_word_index = [None for _ in range(len(example.sentence)+2)] # word -> token
        # for token_i, word_i in enumerate(word_index):
        #     if reverse_word_index[word_i] is None:
        #         reverse_word_index[word_i] = token_i

        if is_training:
            # we need to extend the head to subword level
            token_heads = []
            token_labels = []
            token_label_ids = []
            if has_confidence:
                token_confidence = []
            for wi in word_index:
                if wi == 0: # [CLS]
                    token_heads.append(0)
                    token_labels.append('_')
                    token_label_ids.append(0) # we require the label vocab first position must be '_'
                    if has_confidence:
                        token_confidence.append(0)
                elif wi == len(example.sentence)+1: # [SEP]
                    token_heads.append(0)
                    token_labels.append('_')
                    token_label_ids.append(0)
                    if has_confidence:
                        token_confidence.append(0)
                else:
                    token_heads.append(token_starts[example.heads[wi-1]]) # we set the head to the start of the subwords and we also can use the end
                    token_labels.append(example.labels[wi-1])
                    token_label_ids.append(label_vocab2idx[example.labels[wi-1]])
                    if has_confidence:
                        token_confidence.append(example.confidence[wi-1])
        
        seq_len = len(tokens) - 1 # we need to remove the [SEP]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        if is_training:
            token_heads += padding
            token_label_ids += padding
            if has_confidence:
                token_confidence += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if is_training:
            assert len(token_heads) == max_seq_length
            assert len(token_label_ids) == max_seq_length
            if has_confidence:
                assert len(token_confidence) == max_seq_length

        # if example_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("sent_id: {}".format(example.sent_id))
        #     logger.info("tokens: {}".format(' '.join(tokens)))
        #     logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
        #     logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
        #     logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
        #     if is_training:
        #         logger.info("heads: {}".format(token_heads))
        #         logger.info("labels: {}".format(token_labels))
        #         logger.info("label_ids: {}".format(token_label_ids))
        #         if has_confidence:
        #             logger.info("confidence: {}".format(token_confidence))

        features.append(
            InputFeatures(
                example_id = example_index,
                input_ids = input_ids,
                input_mask = input_mask,
                segment_ids = segment_ids,
                heads = token_heads if is_training else None,
                labels = token_label_ids if is_training else None,
                confidence = token_confidence if is_training and has_confidence else None,
                seq_len = seq_len,
                word_index = word_index,
                token_starts=token_starts,
                token_ends=token_ends,
                example = example
            )
        )
        example_index += 1
    logger.info("dataset size: {} skip {} due to length.".format(len(features), skip_num))
    return features

def load_label_vocab(vocab_path):
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 0:
                vocab.append(line.strip())
    assert vocab[0] == '_'
    vocab2idx = dict()
    for idx,lb in enumerate(vocab):
        vocab2idx[lb] = idx
    return vocab, vocab2idx

def write_conll_examples(words, postags, heads, labels, file_path):
    assert len(words) == len(postags) and len(postags) == len(heads) and len(heads) == len(labels)
    with open(file_path, 'w') as f:
        for i in range(len(words)):
            assert len(words[i]) == len(postags[i]) and len(postags[i]) == len(heads[i]) and len(heads[i]) == len(labels[i])
            for j in range(len(words[i])):
                f.write('{}\t{}\t_\t{}\t_\t_\t{}\t{}\t_\t_\n'.format(j+1, words[i][j], postags[i][j], heads[i][j], labels[i][j]))
            f.write('\n')



def main():
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--train_file",
                        default=None,
                        type=str)
    parser.add_argument("--val_file",
                        default=None,
                        type=str)
    parser.add_argument("--test_file",
                        default=None,
                        type=str)
    parser.add_argument("--test_output",
                        default=None,
                        type=str)
    parser.add_argument("--label_vocab",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--punc_set",
                        default='PU',
                        type=str)
    parser.add_argument("--has_confidence",
                        action='store_true')
    parser.add_argument("--only_save_bert",
                        action='store_true')

    parser.add_argument("--arc_space",
                        default=512,
                        type=int)
    parser.add_argument("--type_space",
                        default=128,
                        type=int)

    parser.add_argument("--log_file",
                        default=None,
                        type=str)

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run predict on the test set.")
    parser.add_argument("--do_greedy_predict",
                        action='store_true',
                        help="Whether to run predict on the test set.")
    parser.add_argument("--do_ensemble_predict",
                        action='store_true',
                        help="Whether to run predict on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for test.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.log_file is None:
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    else:
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        filename=args.log_file,
                        filemode='w',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict and not args.do_greedy_predict and not args.do_ensemble_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        assert args.output_dir is not None

    if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if args.do_train and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    label_vocab, label_vocab2idx = load_label_vocab(args.label_vocab)

    punc_set = set(args.punc_set.split(',')) if args.punc_set is not None else None

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        assert args.train_file is not None
        train_examples = read_conll_examples(args.train_file, is_training = True, has_confidence=args.has_confidence)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.do_train or args.do_predict or args.do_greedy_predict:
        # load the pretrained model
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        model = BertForDependencyParsing.from_pretrained(args.bert_model,
            cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)),
            arc_space=args.arc_space, type_space=args.type_space,
            num_labels=len(label_vocab))

        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)
        
        # 
        parser = model.module if hasattr(model, 'module') else model
    elif args.do_ensemble_predict:
        bert_models = args.bert_model.split(',')
        assert len(bert_models) > 1
        tokenizer = BertTokenizer.from_pretrained(bert_models[0], do_lower_case=args.do_lower_case)
        models = []
        for bm in bert_models:
            model = BertForDependencyParsing.from_pretrained(bm,
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)),
                arc_space=args.arc_space, type_space=args.type_space,
                num_labels=len(label_vocab))
            model.to(device)
            model.eval()
            models.append(model)
        parser = models[0].module if hasattr(models[0], 'module') else models[0]

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        # !!! NOTE why?
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    # start training loop
    if args.do_train:
        global_step = 0
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, label_vocab2idx, True, has_confidence=args.has_confidence)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.float32)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_lengths = torch.tensor([f.seq_len for f in train_features], dtype=torch.long)
        all_heads = torch.tensor([f.heads for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long)

        if args.has_confidence:
            all_confidence = torch.tensor([f.confidence for f in train_features], dtype=torch.float32)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lengths, all_heads, all_labels, all_confidence)
        else:
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lengths, all_heads, all_labels)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.do_eval:
            assert args.val_file is not None
            eval_examples = read_conll_examples(args.val_file, is_training = False, has_confidence=False)
            eval_features = convert_examples_to_features(
                eval_examples, tokenizer, args.max_seq_length, label_vocab2idx, False, has_confidence=False)
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            all_example_ids = torch.tensor([f.example_id for f in eval_features], dtype=torch.long)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.float32)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_lengths = torch.tensor([f.seq_len for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lengths, all_example_ids)

            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        best_uas = 0
        best_las = 0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("Training epoch: {}".format(epoch))
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                if args.has_confidence:
                    input_ids, input_mask, segment_ids, lengths, heads, label_ids, confidence = batch
                else:
                    confidence = None
                    input_ids, input_mask, segment_ids, lengths, heads, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, heads, label_ids, confidence)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step%100 == 0:
                    logger.info("Training loss: {}, global step: {}".format(tr_loss/nb_tr_steps, global_step))

            # we eval every epoch
            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                logger.info("***** Running evaluation *****")

                model.eval()

                eval_predict_words, eval_predict_postags, eval_predict_heads, eval_predict_labels = [],[],[],[]

                for input_ids, input_mask, segment_ids, lengths, example_ids in tqdm(eval_dataloader, desc="Evaluating"):
                    example_ids = example_ids.numpy()

                    batch_words = [eval_features[eid].example.sentence for eid in example_ids]
                    batch_postags = [eval_features[eid].example.postags for eid in example_ids]
                    batch_word_index = [eval_features[eid].word_index for eid in example_ids] # token -> word
                    batch_token_starts = [eval_features[eid].token_starts for eid in example_ids] # word -> token start
                    batch_heads = [eval_features[eid].example.heads for eid in example_ids]


                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    heads = heads.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        # tmp_eval_loss = model(input_ids, segment_ids, input_mask, heads, label_ids)
                        energy = model(input_ids, segment_ids, input_mask)

                    heads_pred, labels_pred = parser.decode_MST(energy.cpu().numpy(), lengths.numpy(), leading_symbolic=0, labeled=True)

                    # we convert the subword dependency parsing to word dependency parsing just the word and token start map
                    pred_heads = []
                    pred_labels = []
                    for i in range(len(batch_word_index)):
                        word_index = batch_word_index[i]
                        token_starts = batch_token_starts[i]
                        hpd = []
                        lpd = []
                        for j in range(len(token_starts)):
                            if j == 0: #[CLS]
                                continue
                            elif j == len(token_starts)-1: # [SEP]
                                continue
                            else:
                                hpd.append(word_index[heads_pred[i, token_starts[j]]])
                                lpd.append(label_vocab[labels_pred[i, token_starts[j]]])
                        pred_heads.append(hpd)
                        pred_labels.append(lpd)

                    eval_predict_words += batch_words
                    eval_predict_postags += batch_postags
                    eval_predict_heads += pred_heads
                    eval_predict_labels += pred_labels
                
                eval_output_file = os.path.join(args.output_dir, 'eval.pred')

                write_conll_examples(eval_predict_words, eval_predict_postags, eval_predict_heads, eval_predict_labels, eval_output_file)

                eval_f = os.popen("python scripts/eval_nlpcc_dp.py "+args.val_file+" "+eval_output_file, "r")
                result_text = eval_f.read().strip()
                logger.info("***** Eval results *****")
                logger.info(result_text)
                eval_f.close()
                eval_res = re.findall(r'UAS = \d+/\d+ = ([\d\.]+), LAS = \d+/\d+ = ([\d\.]+)', result_text)
                assert len(eval_res) > 0
                eval_res = eval_res[0]

                eval_uas = float(eval_res[0])
                eval_las = float(eval_res[1])
                
                # save model
                if best_las < eval_las or (eval_las == best_las and best_uas < eval_uas):
                    best_uas = eval_uas
                    best_las = eval_las

                    logger.info("new best uas  %.2f%% las %.2f%%, saving models.", best_uas, best_las)

                    # Save a trained model, configuration and tokenizer
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                    # If we save using the predefined names, we can load using `from_pretrained`
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                    model_dict = model_to_save.state_dict()
                    if args.only_save_bert:
                        model_dict = {k: v for k, v in model_dict.items() if 'bert.' in k }

                    torch.save(model_dict, output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(args.output_dir)

    # start predict
    if args.do_predict:
        model.eval()
        assert args.test_file is not None
        test_examples = read_conll_examples(args.test_file, is_training = False, has_confidence=False)
        test_features = convert_examples_to_features(
            test_examples, tokenizer, args.max_seq_length, label_vocab2idx, False, has_confidence=False)
        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.test_batch_size)
        all_example_ids = torch.tensor([f.example_id for f in test_features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.float32)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_lengths = torch.tensor([f.seq_len for f in test_features], dtype=torch.long)
        
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lengths, all_example_ids)

        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)
        
        test_predict_words, test_predict_postags, test_predict_heads, test_predict_labels = [],[],[],[]
        for batch_id, batch in enumerate(tqdm(test_dataloader, desc="Predicting")):
            input_ids, input_mask, segment_ids, lengths, example_ids = batch
            example_ids = example_ids.numpy()
            batch_words = [test_features[eid].example.sentence for eid in example_ids]
            batch_postags = [test_features[eid].example.postags for eid in example_ids]
            batch_word_index = [test_features[eid].word_index for eid in example_ids] # token -> word
            batch_token_starts = [test_features[eid].token_starts for eid in example_ids] # word -> token start

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            lengths = lengths.numpy()

            with torch.no_grad():
                energy = model(input_ids, segment_ids, input_mask)

            heads_pred, labels_pred = parser.decode_MST(energy.cpu().numpy(), lengths, leading_symbolic=0, labeled=True)

            pred_heads = []
            pred_labels = []
            for i in range(len(batch_word_index)):
                word_index = batch_word_index[i]
                token_starts = batch_token_starts[i]
                hpd = []
                lpd = []
                for j in range(len(token_starts)):
                    if j == 0: #[CLS]
                        continue
                    elif j == len(token_starts)-1: # [SEP]
                        continue
                    else:
                        hpd.append(word_index[heads_pred[i, token_starts[j]]])
                        lpd.append(label_vocab[labels_pred[i, token_starts[j]]])
                pred_heads.append(hpd)
                pred_labels.append(lpd)

            test_predict_words += batch_words
            test_predict_postags += batch_postags
            test_predict_heads += pred_heads
            test_predict_labels += pred_labels

        assert args.test_output is not None
        write_conll_examples(test_predict_words, test_predict_postags, test_predict_heads, test_predict_labels, args.test_output)

    if args.do_greedy_predict:
        model.eval()
        assert args.test_file is not None
        test_examples = read_conll_examples(args.test_file, is_training = False, has_confidence=False)
        test_features = convert_examples_to_features(
            test_examples, tokenizer, args.max_seq_length, label_vocab2idx, False, has_confidence=False)
        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.test_batch_size)
        all_example_ids = torch.tensor([f.example_id for f in test_features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.float32)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_lengths = torch.tensor([f.seq_len for f in test_features], dtype=torch.long)
        
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lengths, all_example_ids)

        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)
        
        test_predict_words, test_predict_postags, test_predict_heads, test_predict_labels = [],[],[],[]
        for batch_id, batch in enumerate(tqdm(test_dataloader, desc="Predicting")):
            input_ids, input_mask, segment_ids, lengths, example_ids = batch
            example_ids = example_ids.numpy()
            batch_words = [test_features[eid].example.sentence for eid in example_ids]
            batch_postags = [test_features[eid].example.postags for eid in example_ids]
            batch_word_index = [test_features[eid].word_index for eid in example_ids] # token -> word
            batch_token_starts = [test_features[eid].token_starts for eid in example_ids] # word -> token start

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            lengths = lengths.numpy()

            with torch.no_grad():
                heads_pred, labels_pred = model(input_ids, segment_ids, input_mask, greedy_inference=True)

            pred_heads = []
            pred_labels = []
            for i in range(len(batch_word_index)):
                word_index = batch_word_index[i]
                token_starts = batch_token_starts[i]
                hpd = []
                lpd = []
                for j in range(len(token_starts)):
                    if j == 0: #[CLS]
                        continue
                    elif j == len(token_starts)-1: # [SEP]
                        continue
                    else:
                        hpd.append(word_index[heads_pred[i, token_starts[j]]])
                        lpd.append(label_vocab[labels_pred[i, token_starts[j]]])
                pred_heads.append(hpd)
                pred_labels.append(lpd)

            test_predict_words += batch_words
            test_predict_postags += batch_postags
            test_predict_heads += pred_heads
            test_predict_labels += pred_labels

        assert args.test_output is not None
        write_conll_examples(test_predict_words, test_predict_postags, test_predict_heads, test_predict_labels, args.test_output)

    if args.do_ensemble_predict:
        assert args.test_file is not None
        test_examples = read_conll_examples(args.test_file, is_training = False, has_confidence=False)
        test_features = convert_examples_to_features(
            test_examples, tokenizer, args.max_seq_length, label_vocab2idx, False, has_confidence=False)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.test_batch_size)
        all_example_ids = torch.tensor([f.example_id for f in test_features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.float32)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_lengths = torch.tensor([f.seq_len for f in test_features], dtype=torch.long)
        
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lengths, all_example_ids)

        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)
        
        test_predict_words, test_predict_postags, test_predict_heads, test_predict_labels = [],[],[],[]
        for batch_id, batch in enumerate(tqdm(test_dataloader, desc="Predicting")):
            input_ids, input_mask, segment_ids, lengths, example_ids = batch
            example_ids = example_ids.numpy()
            batch_words = [test_features[eid].example.sentence for eid in example_ids]
            batch_postags = [test_features[eid].example.postags for eid in example_ids]
            batch_word_index = [test_features[eid].word_index for eid in example_ids] # token -> word
            batch_token_starts = [test_features[eid].token_starts for eid in example_ids] # word -> token start

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            lengths = lengths.numpy()

            with torch.no_grad():
                energy_sum = None
                for model in models:    
                    energy = model(input_ids, segment_ids, input_mask)
                    if energy_sum is None:
                        energy_sum = energy
                    else:
                        energy_sum = energy_sum + energy
                
                energy_sum = energy_sum / len(models)
            
            heads_pred, labels_pred = parser.decode_MST(energy_sum.cpu().numpy(), lengths, leading_symbolic=0, labeled=True)

            pred_heads = []
            pred_labels = []
            for i in range(len(batch_word_index)):
                word_index = batch_word_index[i]
                token_starts = batch_token_starts[i]
                hpd = []
                lpd = []
                for j in range(len(token_starts)):
                    if j == 0: #[CLS]
                        continue
                    elif j == len(token_starts)-1: # [SEP]
                        continue
                    else:
                        hpd.append(word_index[heads_pred[i, token_starts[j]]])
                        lpd.append(label_vocab[labels_pred[i, token_starts[j]]])
                pred_heads.append(hpd)
                pred_labels.append(lpd)

            test_predict_words += batch_words
            test_predict_postags += batch_postags
            test_predict_heads += pred_heads
            test_predict_labels += pred_labels

        assert args.test_output is not None
        write_conll_examples(test_predict_words, test_predict_postags, test_predict_heads, test_predict_labels, args.test_output)


if __name__ == "__main__":
    main()
