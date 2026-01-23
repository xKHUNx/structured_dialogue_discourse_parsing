import os
import time
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from dataset import SelectionDataset
from model import Model
import pickle


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def eval_running_model(dataloader, test_mode, device, model, args, num_relation_types):
    model.eval()
    
    # Pre-cache input encoding
    pkl_name = f'{test_mode}_{args.test_data_dir}_{args.max_num_test_contexts}_{args.max_contexts_length}_cache.pkl'
    pkl_name = pkl_name.replace('/', '')
    
    if not os.path.exists(pkl_name):
        input_masks, input_types, str_keys = [], [], []
        print('Pre-caching...')
        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch[0].numpy()
            str_keys += [" ".join(item) for item in input_ids.astype(str)]
            input_masks += batch[1].numpy().tolist()
            input_types += batch[2].numpy().tolist()
        mapping = {k: [k, m, t] for k, m, t in zip(str_keys, input_masks, input_types)}
        with open(pkl_name, 'wb') as f:
            pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Loading pre-cached mapping...')
        with open(pkl_name, 'rb') as f:
            mapping = pickle.load(f)
    
    # Encode all inputs
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16):
        print('Encoding...')
        encoder_cache = model.encoder_inference(mapping)
        print('Running inference...')
        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch[0].numpy()
            keys = [" ".join(item) for item in input_ids.astype(str)]
            struct_vec = torch.stack([encoder_cache[key] for key in keys], 0)
            model.inference_forward(struct_vec.to(device), batch[3].to(device), args.max_num_test_contexts)
    
    # Collect results
    tree_results, relation_types = [], []
    for tree_result, predicted_types in model.struct_attention.tree_results:
        tree_results += tree_result
        relation_types += predicted_types
    model.struct_attention.tree_results = []  # clear cache
    
    # Load ground truth
    with open(os.path.join(args.test_data_dir, f'{test_mode}_links.json')) as f:
        gt = json.load(f)

    # Load relation names
    relation_names = {}
    relation_database_path = os.path.join(args.data_dir, 'relation_database.json')
    try:
        with open(relation_database_path) as f:
            relation_db = json.load(f)
        relation_names = {v: k for k, v in relation_db.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: relation database not found or invalid: {relation_database_path}")

    # Save predictions in JSON format
    predictions = []
    for idx, (ds, r) in enumerate(zip(tree_results, relation_types)):
        entry = {
            "id": idx,  # Using simple index since dataset doesn't contain explicit IDs
            "relations": []
        }
        for d in ds:
            for child_idx, parent in enumerate(d[1:]):  # skip root
                # Ensure relation_names is populated before using it
                rel_type_id = r[parent][child_idx+1]
                rel_type_name = relation_names.get(rel_type_id, f'Type_{rel_type_id}')
                entry["relations"].append({
                    "type": rel_type_name,
                    "x": parent,
                    "y": child_idx+1
                })
        predictions.append(entry)
    
    # Write predictions to file
    output_file = os.path.join(args.output_dir, f'{test_mode}_predictions.json')
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Initialize per-relation metrics
    relation_metrics = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(num_relation_types)}
    
    # Initialize micro counters
    tp_total, fp_total, fn_total = 0, 0, 0
    tp_link, fp_link, fn_link = 0, 0, 0

    # Evaluate
    for ds, g, r in zip(tree_results, gt, relation_types):
        all_pred, all_pred_link = set(), set()
        all_gold, all_gold_link = set(), set()

        # Predicted
        for d in ds:
            for idx, dd in enumerate(d[1:]):  # skip root
                pred_type = r[dd][idx+1]
                all_pred.add((dd, idx+1, pred_type))
                all_pred_link.add((dd, idx+1))  # link only

        # Gold
        for gg in g:
            all_gold.add(tuple(gg))
            all_gold_link.add(tuple(gg[:2]))  # link only

        # Update micro-F1 (link+type)
        for triplet in all_pred:
            if triplet in all_gold:
                tp_total += 1
            else:
                fp_total += 1
        for triplet in all_gold:
            if triplet not in all_pred:
                fn_total += 1

        # Update link-only micro-F1
        for link in all_pred_link:
            if link in all_gold_link:
                tp_link += 1
            else:
                fp_link += 1
        for link in all_gold_link:
            if link not in all_pred_link:
                fn_link += 1

        # Update per-relation F1
        for pred_item in all_pred:
            if pred_item in all_gold:
                relation_metrics[pred_item[2]]['tp'] += 1
            else:
                relation_metrics[pred_item[2]]['fp'] += 1
        for gold_item in all_gold:
            if gold_item not in all_pred:
                relation_metrics[gold_item[2]]['fn'] += 1

    # Micro-F1 calculations
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    link_precision = tp_link / (tp_link + fp_link) if (tp_link + fp_link) > 0 else 0
    link_recall = tp_link / (tp_link + fn_link) if (tp_link + fn_link) > 0 else 0
    link_f1 = 2 * link_precision * link_recall / (link_precision + link_recall) if (link_precision + link_recall) > 0 else 0

    results = {
        'micro_f1': f1,
        'micro_precision': precision,
        'micro_recall': recall,
        'link_f1': link_f1,
        'link_precision': link_precision,
        'link_recall': link_recall
    }

    # Per-relation F1
    for rel_id, metrics in relation_metrics.items():
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        rel_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rel_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        rel_f1 = 2 * rel_prec * rel_rec / (rel_prec + rel_rec) if (rel_prec + rel_rec) > 0 else 0
        rel_name = relation_names.get(rel_id, f'Type_{rel_id}')
        results[f'{rel_name}_f1'] = rel_f1
        results[f'{rel_name}_precision'] = rel_prec
        results[f'{rel_name}_recall'] = rel_rec

    return results

def evaluate(args, epoch, global_step, dev_dataloader, test_dataloader, best_f1, model, device, num_relation_types):
    dev_result = eval_running_model(dev_dataloader, 'dev', device, model, args, num_relation_types)
    test_result = eval_running_model(test_dataloader, 'test', device, model, args, num_relation_types)
    print('Epoch %d, Global Step %d DEV res:\n' % (epoch, global_step), dev_result)
    print('Epoch %d, Global Step %d TST res:\n' % (epoch, global_step), test_result)
    log_wf.write('Global Step %d VAL res:\n' % global_step)
    log_wf.write('Global Step %d TST res:\n' % global_step)
    log_wf.write(str(dev_result) + '\n')
    log_wf.write(str(test_result) + '\n')
    # save model
    if dev_result['micro_f1'] > best_f1:
        # save model
        state_save_path = os.path.join(args.output_dir, 'pytorch_model.bin')
        print('[Saving at]', state_save_path)
        log_wf.write('[Saving at] %s\n' % state_save_path)
        torch.save(model.state_dict(), state_save_path)

    return max(best_f1, dev_result['micro_f1'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--encoder_model", required=True, type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--output_dir", default='/dev/null', type=str)
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--test_data_dir", type=str)

    parser.add_argument("--max_contexts_length", default=28, type=int, help="Number of tokens per context")
    parser.add_argument("--max_num_train_contexts", type=int, help="Number of train contexts")
    parser.add_argument("--max_num_dev_contexts", type=int, help="Number of dev contexts")
    parser.add_argument("--max_num_test_contexts", type=int, help="Number of test contexts")
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int, help="Total batch size for eval.")
    parser.add_argument("--print_freq", default=100, type=int, help="Log frequency")
    parser.add_argument("--link_only", action="store_true")
    parser.add_argument("--cross_domain", action="store_true")

    parser.add_argument("--use_scheduler", action="store_true", help='Whether to use scheduler for learning rate adjustment')
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient Accumulation Step")
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Warmup optimization steps percentage")
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta_1", default=0.9, type=float, help="beta_1 for Adam optimizer")
    parser.add_argument("--beta_2", default=0.999, type=float, help="beta_2 for Adam optimizer")
    parser.add_argument("--max_grad_norm", default=float('inf'), type=float, help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                                            help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision instead of 32-bit",
    )
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.test_data_dir is None:
        args.test_data_dir = args.data_dir
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args)

    # Determine num_types from relation_database.json
    relation_database_path = os.path.join(args.data_dir, 'relation_database.json')
    num_relation_types = 17 # Default value
    try:
        with open(relation_database_path) as f:
            relation_database = json.load(f)
        if not relation_database:
            print(f"Warning: {relation_database_path} is empty or invalid. Using default num_types={num_relation_types}.")
        else:
            num_relation_types = len(list(relation_database.values()))
        print(f"Determined num_types from {relation_database_path}: {num_relation_types}")
    except FileNotFoundError:
        print(f"Error: {relation_database_path} not found. Using default num_types={num_relation_types}.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {relation_database_path}. Using default num_types={num_relation_types}.")
    except ValueError: # Handles max() on empty sequence if relation_database was empty and not caught by 'if not relation_database'
        print(f"Error: relation_database at {relation_database_path} is empty or has invalid values. Using default num_types={num_relation_types}.")


    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    if not args.eval:
        train_dataset = SelectionDataset(os.path.join(args.data_dir, 'train.txt'), args, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.batchify_join_str, shuffle=True, num_workers=1)
        t_total = len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
        dev_dataset = SelectionDataset(os.path.join(args.test_data_dir, 'dev.txt'), args, tokenizer)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=dev_dataset.batchify_join_str, shuffle=False, num_workers=1)
    test_dataset = SelectionDataset(os.path.join(args.test_data_dir, 'test.txt'), args, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.batchify_join_str, shuffle=False, num_workers=1)

    encoder_config = AutoConfig.from_pretrained(os.path.join(args.encoder_model, 'config.json'))
    if not args.eval:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        log_wf = open(os.path.join(args.output_dir, 'log.txt'), 'a')
        shutil.copy(os.path.join(args.encoder_model, 'config.json'), args.output_dir)
        shutil.copy(os.path.join(args.encoder_model, 'tokenizer.json'), args.output_dir)
        encoder = AutoModel.from_pretrained(args.encoder_model)
    else:
        encoder = AutoModel.from_config(encoder_config)

    # Modify the model initialization part
    if args.eval:
        # When evaluating, always load model with full relation types regardless of link_only flag
        model = Model(encoder_config, encoder=encoder, link_only=False, num_types=num_relation_types).to(device)
        state_save_path = os.path.join(args.encoder_model, 'pytorch_model.bin')
        print('Loading parameters from', state_save_path)
        model.load_state_dict(torch.load(state_save_path, map_location=torch.device('cpu')))
        test_result = eval_running_model(test_dataloader, 'test', device, model, args, num_relation_types)
        print(test_result)
        exit()
    else:
        # For training, use link_only as specified
        model = Model(encoder_config, encoder=encoder, link_only=args.link_only, num_types=num_relation_types).to(device)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta_1, args.beta_2), eps=args.adam_epsilon)
    if args.use_scheduler:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(t_total*args.warmup_ratio), num_training_steps=t_total
        )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    global_step = 0
    best_f1 = 0
    
    for epoch in range(1, int(args.num_train_epochs) + 1):
        tr_loss = 0
        nb_tr_steps = 0
        with tqdm(total=len(train_dataloader)//args.gradient_accumulation_steps) as bar:
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    loss = model(*batch, max_sent_len=args.max_num_train_contexts)
                    loss = loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    nb_tr_steps += 1
                    scaler.step(optimizer)
                    scaler.update()
                    if args.use_scheduler:
                        scheduler.step()
                    model.zero_grad()
                    optimizer.zero_grad()
                    global_step += 1

                    if nb_tr_steps and nb_tr_steps % args.print_freq == 0:
                        bar.update(min(args.print_freq, nb_tr_steps))
                        time.sleep(0.02)
                        print(global_step, tr_loss / nb_tr_steps)
                        log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))
                        if args.cross_domain:
                            best_f1 = evaluate(args, epoch, global_step, dev_dataloader, test_dataloader, best_f1, model, device, num_relation_types)
        
        best_f1 = evaluate(args, epoch, global_step, dev_dataloader, test_dataloader, best_f1, model, device, num_relation_types)
