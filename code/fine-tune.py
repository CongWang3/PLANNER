import argparse
import glob
import json
import logging
import os
import re
import pickle
import shutil
import random
from multiprocessing import Pool
from typing import Dict, List, Tuple
from copy import deepcopy
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, roc_auc_score
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

def load_model(args):
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        num_labels=num_labels,finetuning_task=args.task_name,
                                        cache_dir=None,)
    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    if args.model_type in ["dnalong", "dnalongcat"]:
        assert args.max_seq_length % 512 == 0
    config.rnn = args.rnn
    config.num_rnn_layer = args.num_rnn_layer
    config.rnn_dropout = args.rnn_dropout
    config.rnn_hidden = args.rnn_hidden
        

    model = model_class.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
    cache_dir=None,).to(args.device)
    print('finish loading model', args.model_name_or_path)
    
    tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir= None,
        )
    print('finish loading tokenizer', args.tokenizer_name)
    return model, tokenizer


logger = logging.getLogger(__name__)
def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (  
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )   

        
        print("finish loading examples")

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
                    

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

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

def train(args, train_dataset, model, tokenizer, sp = 'A'):  # index用于args.model_name_or_path
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer_1 = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloader_1 = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False)
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader_1) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader_1) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters_1 = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent*t_total)
    optimizer_1 = AdamW(optimizer_grouped_parameters_1, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1,args.beta2))
    scheduler_1 = get_linear_schedule_with_warmup(
        optimizer_1, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Train!
    print("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader_1) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader_1) // args.gradient_accumulation_steps)

    tr_loss_1, logging_loss_1 = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    
    best_auc = 0
    last_auc = 0
    stop_count = 0

    for _ in train_iterator:
        train_bar = tqdm(train_dataloader_1, desc="Iteration", disable=args.local_rank not in [-1, 0])
        preds_1 = None
        for step, batch_1 in enumerate(train_bar):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch_1 = tuple(t.to(args.device) for t in batch_1)
            inputs_1 = {"input_ids": batch_1[0], "attention_mask": batch_1[1], "labels": batch_1[3]}
            if args.model_type != "distilbert":
                inputs_1["token_type_ids"] = (
                    batch_1[2] if args.model_type in TOKEN_ID_GROUP else None
                )
            outputs_1 = model(**inputs_1)
            loss_1 = outputs_1[0]
            logits_1 = outputs_1[1]

            if preds_1 is None:
                preds_1 = logits_1.detach().cpu().numpy()
                out_label_ids_1 = inputs_1["labels"].detach().cpu().numpy()
            else:
                preds_1 = np.append(preds_1, logits_1.detach().cpu().numpy(), axis=0)
                out_label_ids_1 = np.append(out_label_ids_1, inputs_1["labels"].detach().cpu().numpy(), axis=0)
            

            if args.n_gpu > 1:
                loss_1 = loss_1.mean()
            if args.gradient_accumulation_steps > 1:
                loss_1 = loss_1 / args.gradient_accumulation_steps
                
            loss_1.backward()
            tr_loss_1 += loss_1.item()

            # evaluate_during_training, early stop, save model
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # print('evaluate_during_training, early stop, save model')
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer_1.step()
                scheduler_1.step()  # Update learning rate schedule
                model.zero_grad()            
                global_step += 1
                
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    results = {}
                    if (args.local_rank == -1 and args.evaluate_during_training):
                        print('evaluate during train, data_dir')
                        results = evaluate(args, model, tokenizer, evaluate = True, cell=sp)

                        if args.task_name == "dna690":
                            # record the best auc
                            if results["auc"] > best_auc:
                                best_auc = results["auc"]
                        if args.early_stop != 0:
                            # record current auc to perform early stop
                            if results["auc"] < last_auc:
                                stop_count += 1
                            else:
                                stop_count = 0

                            last_auc = results["auc"]
                            
                            if stop_count == args.early_stop:
                                logger.info("Early stop")
                                return model, tokenizer, preds_1, out_label_ids_1, global_step, tr_loss_1 / global_step
                    
                    for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss_1 - logging_loss_1) / args.logging_steps
                    learning_rate_scalar = scheduler_1.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss_1 = tr_loss_1
                    
                    for key, value in logs.items():
                        tb_writer_1.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))
              
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    print('inner save model global step =', global_step)
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    print("Saving inner model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(optimizer_1.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler_1.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            
            if args.max_steps > 0 and global_step > args.max_steps:
                train_bar.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer_1.close()

    return model, tokenizer, global_step, tr_loss_1 / global_step


def evaluate(args, model, tokenizer, prefix="", evaluate=True, cell='A'): 
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)
    if args.task_name[:3] == "dna":
        softmax = torch.nn.Softmax(dim=1)
        
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=evaluate)
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
                out_label_ids_i = inputs["labels"].detach().cpu().numpy()
            else:
                preds_i = np.append(preds_i, logits.detach().cpu().numpy(), axis=0)
                out_label_ids_i = np.append(out_label_ids_i, inputs["labels"].detach().cpu().numpy(), axis=0)

        true_label = out_label_ids_i

        probs_i = softmax(torch.tensor(preds_i, dtype=torch.float32)).numpy()
        predict_label = np.argmax(probs_i, axis=1)

        result = compute_metrics(eval_task, predict_label, true_label, probs_i[:, 1])
        results.update(result)

        save_path = args.output_dir+'/result.log'
        with open(save_path, mode='w') as file:
            r = ''
            for key in result.keys():
                r += key + ' =' + str(round(result[key], 4))  + ', '
            print(cell+' =',r, '\n')
            file.write(r)
        
        return results



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="dna")
    parser.add_argument("--tokenizer_name", default="dna3")
    parser.add_argument("--model_name_or_path", default="./examples/DNAbert_3mer")
    parser.add_argument("--max_seq_length", default=300)
    parser.add_argument("--task_name", default="dnaprom")
    parser.add_argument("--output_mode", default="classification")
    parser.add_argument("--config_name", default="")
    parser.add_argument("--early_stop", default=2)
    parser.add_argument("--hidden_dropout_prob", default=0.1)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1)
    parser.add_argument("--rnn", default="lstm")
    parser.add_argument("--num_rnn_layer", default=2)
    parser.add_argument("--rnn_dropout", default=0.0)
    parser.add_argument("--rnn_hidden", default=768)
    parser.add_argument("--local_rank", default=-1)
    parser.add_argument("--n_process", default=8)
    parser.add_argument("--per_gpu_train_batch_size", default=32)
    parser.add_argument("--overwrite_cache", default=True)
    parser.add_argument("--per_gpu_eval_batch_size", default=32)
    parser.add_argument("--train_batch_size", default=1)
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--max_steps", default=-1)
    parser.add_argument("--gradient_accumulation_steps", default=1)
    parser.add_argument("--num_train_epochs", default=1.0)
    parser.add_argument("--warmup_steps", default=0)
    parser.add_argument("--warmup_percent", default=0.1)
    parser.add_argument("--learning_rate", default=5e-5)
    parser.add_argument("--beta1", default=0.9)
    parser.add_argument("--beta2", default=0.999)
    parser.add_argument("--adam_epsilon", default=1e-8)
    parser.add_argument("--weight_decay", default=0.01)
    parser.add_argument("--device", default=0)
    parser.add_argument("--max_grad_norm", default=1.0)
    parser.add_argument("--fp16", default=False)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--logging_steps", default=500)
    parser.add_argument("--save_steps", default=4000)
    parser.add_argument("--save_total_limit", default=None)
    parser.add_argument("--do_ensemble_pred", default=False)
    parser.add_argument("--evaluate_during_training", default=False)
    parser.add_argument("--do_lower_case", default=False)
    parser.add_argument("--data_dir", default="./examples/sample_data/ORI/3/", help="File save path for training and testing data, with the training file name train.tsv and testing file name dev.tsv")
    parser.add_argument("--output_dir", help='Save path for results')
    
    args = parser.parse_args()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    print('args.device =', args.device)
    set_seed(args)
    
    species = ['A']  # ['A', 'H', 'M', 'D']
    for sp in species:
        if sp == 'D':
            cells = ['D-Bg3', 'D-S2', 'D-Kc']
        elif sp == 'H':
            cells = ['H-MCF7', 'H-K562']
        elif sp == 'M':
            cells = ['M-ES', 'M-P19', 'M-MEF']
        else:
            cells = ['A']
          
        model, tokenizer = load_model(args)
        if os.path.exists(args.output_dir) == False:
            os.makedirs(args.output_dir)

        train_dataset_1 = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)

        model, tokenizer, global_step, tr_loss = train(args, train_dataset_1, model, tokenizer, sp=sp)
        print('global_step =', round(global_step, 5), 'average loss =', round(tr_loss, 5), 'species =', sp)
        
        checkpoint_prefix = "checkpoint"
        name = 'checkpoint'
        output_dir = os.path.join(args.output_dir, name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print("Saving model out checkpoint" , " to ", str(output_dir))

        _rotate_checkpoints(args, checkpoint_prefix)
            
        for cell in cells:
            result = evaluate(args, model, tokenizer, evaluate=True, cell=cell)


    
