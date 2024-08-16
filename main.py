import sys
sys.path.insert(1, '../src')
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset

import json
from tqdm import tqdm
from datetime import datetime
import copy
import argparse
import re
import random

from utils import *
from evaluate import evaluate_sepsis_score
from config import *
from models import *
from dataset import *
from prepare import train_test_split


def build_dataset(dataset, model_type, data_config, downsample):
    window_size = data_config["window_size"]
    start_offset = data_config["start_offset"]
    columns = data_config["columns"]
    
    if dataset == 'synthetic':
        with open(synthetic_train_ids_filepath, "r") as f:
            train_ids = json.load(f)
    
        with open(synthetic_test_ids_filepath, "r") as f:
            test_ids = json.load(f)

        dataset = SyntheticDataset(train_ids, window_size, start_offset, columns)
        testset = SyntheticDataset(test_ids, window_size, start_offset,columns)
    else:
        if not (os.path.exists(train_ids_filepath) and os.path.exists(test_ids_filepath)):
            train_test_split()
            
        with open(train_ids_filepath, "r") as f:
            train_ids = json.load(f)
    
        with open(test_ids_filepath, "r") as f:
            test_ids = json.load(f)

        if downsample != 1:
            with open(sepsis_ids_filepath, "r") as fp:
                sepsis_ids = json.load(fp)
            train_pos = [value for value in train_ids  if value in sepsis_ids]
            train_neg = list(set(train_ids) - set(train_pos))
            random.seed(42)
            train_neg = random.sample(train_neg, int(len(train_neg)*downsample))
            train_ids = train_pos + train_neg
            train_ids.sort()
    
        if not model_type == 'WeibullCox':
            dataset = SepsisDataset(train_ids, window_size, start_offset, columns)
            testset = SepsisDataset(test_ids, window_size, start_offset, columns)
            print(f'Composition of datasets: train {dataset.get_ratio()} test {testset.get_ratio()}')
        else:
            dataset = WeibullCoxDataset(train_ids)
            testset = RawDataset(test_ids)
    
    return dataset, testset


def build_model(model_type, config, model_path):
    assert model_type == config['model']['type'], "Model type not match! Check config!"
    
    input_shape = config['model']['input_shape']
    assert config['data']['window_size'] == input_shape[0], "Window size not match! Check config!"
    
    if model_type == 'Log':
        model = LogisticRegressionModel(input_shape, 1, config, model_path)
    elif model_type == 'MLP':
        model = MLPModel(input_shape, 1, config, model_path)
    elif model_type == 'RNN':
        model = RNNModel(config, model_path)
    elif model_type == 'LSTM':
        model = LSTMModel(config, model_path)
    elif model_type == 'GRU':
        model = GRUModel(config, model_path)
    elif model_type == 'ResNet':
        model = ResNetModel(input_shape, 1, config, model_path)
    elif model_type == 'Transformer':
        model = TransformerModel(input_shape, 1, config, model_path)
    elif model_type == 'WeibullCox':
        if window_size > 1:
            raise Exception("Weibull-Cox model got window size > 1!")
            sys.exit()
        model = WeibullCoxModel(model_path)

    print('Model has {} parameters'.format(sum(p.numel() for p in model.parameters())))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sepsis binary classification training.")
    
    parser.add_argument('--model-type', type=str, required=True, help='Model architecture')
    parser.add_argument('--config-path', type=str, default='model_config.json', help='Model configuration file path')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--pos-weight', type=float, default=54.5, help='Positive class weight (default: 54.5)')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs for training (default: 25)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--load-saved', action='store_true', help='Flag to load saved model path specified in config')
    parser.add_argument('--skip-eval', action='store_true', help='Flag to skip model evaluation after trainig')
    parser.add_argument('--skip-train', action='store_true', help='Flag to skip training')
    parser.add_argument('--use-val', action='store_true', help='Flag to enable train validation split')
    parser.add_argument('--dataset', type=str, default='physionet', help='Model architecture')
    parser.add_argument('--logging', action='store_true', help='Flag to log to wandb')
    parser.add_argument('--num-workers', type=int, default=24, help='Number of workers for dataloader (default 24)')
    parser.add_argument('--loss', type=str, default='BCE', help='Loss criterion, default BCE')
    parser.add_argument('--downsample', type=float, default=1, help='Rate to downsample negative class in training set (e.g. 0.1)')
    
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    dataset, testset = build_dataset(
        args.dataset, 
        args.model_type, 
        config['data'],
        args.downsample
    )

    if args.load_saved:
        try:
            model_path = config['model'].pop('saved_path')
        except Exception as err:
            print(f"Error in model path config {err=}, {type(err)=}")
    else:
        model_path = None
    
    model = build_model(
        args.model_type, 
        config, 
        model_path
    )

    if not args.skip_train:
        rid = model.train_model(
            dataset, 
            args.use_val, 
            args.epochs, 
            args.batch_size, 
            args.pos_weight, 
            args.learning_rate,
            args.loss,
            args.logging,
            args.num_workers
        )
    elif model_path:
        pattern = r'[^/]+/[^/]+/([^_]+_[^_]+_[^_]+)'
        match = re.search(pattern, model_path)
        if match:
            rid = match.group(1)
            print('Using saved model with rid {}'.format(rid))
        else:
            print('Error loading saved model {}'.format(model_path))
            sys.exit()
    else:
        rid = args.model_type + '_trial'
        print('Using initialized model')

    if not args.skip_eval:
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        results, y_label, y_prob = evaluate_model(model, rid, test_loader)
        cutoff = plot_curves(rid+'_test', y_label, y_prob)
        y_label, y_pred, pred_dirpath = save_pred(results, cutoff, rid)
        print('Using cutoff {}'.format(cutoff))
        print_confusion_matrix(y_label, y_pred)
        auroc, auprc, accuracy, f_measure, utility = evaluate_sepsis_score(label_dirpath, pred_dirpath)
        print('AUROC     {:.4f} \nAUPRC     {:.4f} \nAccuracy  {:.4f} \nF-measure {:.4f} \nUtility   {:.4f}\n'.format(auroc, auprc, accuracy, f_measure, utility))
