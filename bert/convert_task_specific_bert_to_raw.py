# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert BERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import argparse
import torch
import numpy as np
import shutil
import sys

sys.path.append('.')

from bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from bert.tokenization import VOCAB_NAME

def convert_task_specific_bert_to_raw(model_path, save_path):
    model_file = os.path.join(model_path, WEIGHTS_NAME)
    model_dict = torch.load(model_file)
    model_dict = {k: v for k, v in model_dict.items() if 'bert.' in k }
    output_model_file = os.path.join(save_path, WEIGHTS_NAME)
    torch.save(model_dict, output_model_file)

    shutil.copyfile(os.path.join(model_path, CONFIG_NAME),os.path.join(save_path, CONFIG_NAME))
    shutil.copyfile(os.path.join(model_path, VOCAB_NAME),os.path.join(save_path, VOCAB_NAME))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--model_path",
                        default = None,
                        type = str,
                        required = True)
    parser.add_argument("--save_path",
                        default = None,
                        type = str,
                        required = True)
    args = parser.parse_args()
    convert_task_specific_bert_to_raw(args.model_path,
                                     args.save_path)
