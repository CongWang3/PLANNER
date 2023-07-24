import argparse
import glob
import json
import logging
import pandas as pd
import os
import re
import pickle
import shutil
import random
from operator import itemgetter
from multiprocessing import Pool
from typing import Dict, List, Tuple
from copy import deepcopy
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import itertools
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertForLongSequenceClassification,
    BertForLongSequenceClassificationCat,
    BertTokenizer,
    DNATokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

MODEL_CLASSES = {
    "dna": (BertConfig, BertForSequenceClassification, DNATokenizer),
    "dnalong": (BertConfig, BertForLongSequenceClassification, DNATokenizer),
    "dnalongcat": (BertConfig, BertForLongSequenceClassificationCat, DNATokenizer),
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
            FlaubertConfig,
        )
    ),
    (),
)

TOKEN_ID_GROUP = ["bert", "dnalong", "dnalongcat", "xlnet", "albert"] 



class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value
    
def load_one_model(args, model_data_dic, kmer):
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = []
    model = []

    config = config_class.from_pretrained(args.config_name if args.config_name else model_data_dic['model_name_or_path' + kmer],
                                        num_labels=num_labels,finetuning_task=args.task_name,
                                        cache_dir=None,from_tf=True,)

    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    if args.model_type in ["dnalong", "dnalongcat"]:
        assert args.max_seq_length % 512 == 0
    config.rnn = args.rnn
    config.num_rnn_layer = args.num_rnn_layer
    config.rnn_dropout = args.rnn_dropout
    config.rnn_hidden = args.rnn_hidden

    model = model_class.from_pretrained(
    model_data_dic['model_name_or_path' + kmer],
    from_tf=bool(".ckpt" in model_data_dic['model_name_or_path' + kmer]),
    config=config,
    cache_dir=None,).to(args.device)
    print('finish loading model')
    
    tokenizer = tokenizer_class.from_pretrained(
        model_data_dic['tokenizer_name' + kmer] if model_data_dic['tokenizer_name' + kmer] else model_data_dic['model_name_or_path' + kmer],
        do_lower_case=args.do_lower_case,
        cache_dir = None,
    )

    print('finish loading tokenizers')
    return model, tokenizer

logger = logging.getLogger(__name__)

def load_and_cache_examples(args, model_data_dic, task, tokenizer, evaluate=True, index = '_3'):
    processor = processors[task]()
    output_mode = output_modes[task]

    label_list = processor.get_labels()
    examples = (processor.get_dev_examples(model_data_dic['data_dir' + index]))
    
    print("finish loading examples" + index)

    # params for convert_examples_to_features
    max_length = args.max_seq_length
    pad_on_left = bool(args.model_type in ["xlnet"])
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0

    if args.n_process == 1:
        features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=max_length,
        output_mode=output_mode,
        pad_on_left=pad_on_left,  # pad on the left for xlnet
        pad_token=pad_token,
        pad_token_segment_id=pad_token_segment_id,)
            
    else:
        n_proc = int(args.n_process)
        if evaluate:
            n_proc = max(int(n_proc/4),1)
        print("number of processes for converting feature: " + str(n_proc))
        p = Pool(n_proc)
        indexes = [0]
        len_slice = int(len(examples)/n_proc)
        for i in range(1, n_proc+1):
            if i != n_proc:
                indexes.append(len_slice*(i))
            else:
                indexes.append(len(examples))
        
        results = []
        
        for i in range(n_proc):
            results.append(p.apply_async(convert_examples_to_features, args=(examples[indexes[i]:indexes[i+1]], tokenizer, max_length, None, label_list, output_mode, pad_on_left, pad_token, pad_token_segment_id, True,  )))
        print(str(n_proc) + ' processor started !')
        
        p.close()
        p.join()

        features = []
        for result in results:
            features.extend(result.get())

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def evaluate(args, model_data_dic, model, tokenizer, prefix="", evaluate=True, cell='A', kmer='_3'):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = 'dnaprom'
    if args.task_name[:3] == "dna":
        softmax = torch.nn.Softmax(dim=1)
        
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset = load_and_cache_examples(args, model_data_dic, eval_task, tokenizer, evaluate=evaluate, index = kmer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds_i,  probs_i, out_label_ids_i = None, None, None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in TOKEN_ID_GROUP else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds_i is None:
            preds_i = logits.detach().cpu().numpy()
        else:
            preds_i = np.append(preds_i, logits.detach().cpu().numpy(), axis=0)
 
    probs_i = softmax(torch.tensor(preds_i, dtype=torch.float32)).numpy()[:, 1]
    feature = probs_i.reshape((len(probs_i), 1))
    return feature

def file2kmer(filePath):
    file = open(filePath)
    seq_name = []
    final_3mer = []
    final_4mer = []
    final_5mer = []
    final_6mer = []

    for line in file.readlines():
        if line[0] == '>':
            seq_name.append(line[1: -1])
            # print(seq_name)
        else:
            seq = line.strip()
            kmers_3 = [seq[x:x+3] for x in range(len(seq)+1-3)]
            kmers_3 = " ".join(kmers_3)
            kmers_3 = kmers_3 + ' 0\n'
            final_3mer.append(kmers_3)

            kmers_4 = [seq[x:x+4] for x in range(len(seq)+1-4)]
            kmers_4 = " ".join(kmers_4)
            kmers_4 = kmers_4 + ' 0\n'
            final_4mer.append(kmers_4)

            kmers_5 = [seq[x:x+5] for x in range(len(seq)+1-5)]
            kmers_5 = " ".join(kmers_5)
            kmers_5 = kmers_5 + ' 0\n'
            final_5mer.append(kmers_5)

            kmers_6 = [seq[x:x+6] for x in range(len(seq)+1-6)]
            kmers_6 = " ".join(kmers_6)
            kmers_6 = kmers_6 + ' 0\n'
            final_6mer.append(kmers_6)
        
    for k in [3,4,5,6]:
        path = './examples/sample_data/ORI/pipeline/' +'/'+str(k)
        if os.path.exists(path) == False:
            os.makedirs(path)
            
    with open('./examples/sample_data/ORI/pipeline/' + str(3) + '/dev.tsv', mode = 'w') as file_3mer:
        file_3mer.write('sequence	label\n')
        for line in final_3mer:
            file_3mer.write(line)
    with open('./examples/sample_data/ORI/pipeline/' + str(4) + '/dev.tsv', mode = 'w') as file_4mer:
        file_4mer.write('sequence	label\n')
        for line in final_4mer:
            file_4mer.write(line)
    with open('./examples/sample_data/ORI/pipeline/' + str(5) + '/dev.tsv', mode = 'w') as file_5mer:
        file_5mer.write('sequence	label\n')
        for line in final_5mer:
            file_5mer.write(line)
    with open('./examples/sample_data/ORI/pipeline/' + str(6) + '/dev.tsv', mode = 'w') as file_6mer:
        file_6mer.write('sequence	label\n')
        for line in final_6mer:
            file_6mer.write(line)

    return seq_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="examples/sample_data/ORI/pipeline/seq.fasta", type=str, help="The input data file")
    parser.add_argument("--cell_type", default="A", type=str, required=True, help="The ORIs of specific cell that you want to predict.")
    parser.add_argument("--seq", default="")
    parser.add_argument("--model_type", default="dna")
    parser.add_argument("--max_seq_length", default=300)
    parser.add_argument("--task_name", default="dnaprom")
    parser.add_argument("--output_mode", default="classification")
    parser.add_argument("--config_name", default="")
    parser.add_argument("--do_lower_case", default=False)
    parser.add_argument("--hidden_dropout_prob", default=0.1)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1)
    parser.add_argument("--rnn", default="lstm")
    parser.add_argument("--num_rnn_layer", default=2)
    parser.add_argument("--rnn_dropout", default=0.0)
    parser.add_argument("--rnn_hidden", default=768)
    parser.add_argument("--n_process", default=8)
    parser.add_argument("--per_gpu_eval_batch_size", default=8)
    parser.add_argument("--eval_batch_size", default=8)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--device", default=0)
    parser.add_argument("--output_file", default="examples/sample_data/ORI/pipeline/result.csv")


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print('args.device =', args.device)
    

    filePath = args.input_file
    print(filePath)
    seq_name = file2kmer(filePath)
    cell = args.cell_type
    print(cell)

    if cell in ['M-ES', 'M-P19', 'D-Kc', 'D-Bg3', 'D-S2','M-MEF']:
        kmer = ['_3', '_4', '_5', '_6']
    elif cell in ['H-K562', 'A']:
        kmer = ['_3', '_4', '_6']
    elif cell in ['H-MCF7']:
        kmer = ['_3', '_5']
    else:
        kmer = []  # 3456
    sp = cell[0]

    model_data_dic = {}
    model_data_dic = DotDict(model_data_dic)
    model_data_dic['model_name_or_path_3'] = './examples/sample_data/ORI/'+sp+'/3/model/'
    model_data_dic['model_name_or_path_4'] = './examples/sample_data/ORI/'+sp+'/4/model/'
    model_data_dic['model_name_or_path_5'] = './examples/sample_data/ORI/'+sp+'/5/model/'
    model_data_dic['model_name_or_path_6'] = './examples/sample_data/ORI/'+sp+'/6/model/'
    
    model_data_dic['tokenizer_name_3'] = 'dna3'
    model_data_dic['tokenizer_name_4'] = 'dna4'
    model_data_dic['tokenizer_name_5'] = 'dna5'
    model_data_dic['tokenizer_name_6'] = 'dna6'

    model_data_dic['data_dir_3'] = './examples/sample_data/ORI/pipeline/3'
    model_data_dic['data_dir_4'] = './examples/sample_data/ORI/pipeline/4'
    model_data_dic['data_dir_5'] = './examples/sample_data/ORI/pipeline/5'
    model_data_dic['data_dir_6'] = './examples/sample_data/ORI/pipeline/6'

    feature = None
    for k in kmer:
        print(k)
        model, tokenizer = load_one_model(args, model_data_dic, k)
        tem_feature = evaluate(args, model_data_dic, model, tokenizer, prefix="", evaluate=True, cell=cell, kmer=k)
        if feature is None:
            feature = tem_feature
        else:
            feature = np.hstack((feature, tem_feature.reshape((len(tem_feature), 1))))

    # ensemble
    if cell in ['M-MEF']:
        with open('examples/sample_data/ORI/model/M/lr.pkl', mode='rb') as file:
            lr = pickle.load(file)
        label = lr.predict(feature)
        prob = lr.predict_proba(feature)[:, 1]
        print('label.shape =', label.shape)
        for index, name in enumerate(seq_name):
            print(seq_name[index] ,': ', label[index])
    else:
        prob = None
        for i in range(0, len(kmer)):
            if prob is None:
                prob = feature[:, i]
            else:
                prob += feature[:, i]
        prob = prob / len(kmer)
        print('prob.shape =', prob.shape)
        print('prob =', prob)
        label = []
        for index, name in enumerate(seq_name):
            if prob[index] < 0.5:
                label.append(0)
            else:
                label.append(1)
            print(seq_name[index] ,': ', label[index])
            
    print('End')
    result = {'seq_name': seq_name, 'probability': prob ,'label': label}
    result_df = pd.DataFrame(result)
    print("output_file path = ", args.output_file)
    result_df.to_csv(args.output_file, index=False)
    


    


 
