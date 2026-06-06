import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from ctc_decoder import *
from config import *
from dataset import PlateDataset, plate_collate_fn
from model import CRNN, CRNN_WithComplexityHead

import cv2
import torch
import torch.optim as optim
from torch.nn import CTCLoss

from rapidfuzz.distance.Levenshtein import distance 
import copy

import random
from time import time

from peft import get_peft_model, LoraConfig, TaskType
import loralib as lora

import wandb

def set_global_seed(seed):
    """
    Sets random seeds everywhere 
    Args:
        seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(57)

class Timer():
    def __init__(self, name: str, notify: bool = True):
        self.name = name
        self.notify = notify
        self.start_time = time()
    def __enter__(self):
        self.start_time = time()
        if self.notify:
            print(f'> "{self.name}" started')
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time()
        self.elapsed_time = self.end_time - self.start_time
        if self.notify:
            print(f'< {str(self)}\n')
    def __str__(self):
        return f'"{self.name}" took {self.elapsed_time:.2f} seconds'


device = 1


# make dataloaders
train_ds = PlateDataset('/mnt/calc/lizzzafomenko/rcnn/license_plates', 'train')
valid_ds = PlateDataset('/mnt/calc/lizzzafomenko/rcnn/license_plates', 'valid') 

train_config = train_config
evaluate_config = evaluate_config

train_dataloader = DataLoader(
        train_ds, 
        batch_size = train_config['train_batch_size'], 
        shuffle=True, 
        num_workers=2, 
        collate_fn=plate_collate_fn,
        drop_last=True
)

val_dataloader = DataLoader(
        valid_ds, 
        batch_size = train_config['eval_batch_size'], 
        shuffle=False, 
        num_workers=2, 
        collate_fn=plate_collate_fn,
        drop_last=False
)


test_ds = PlateDataset('/mnt/calc/lizzzafomenko/rcnn/license_plates', 'test')
evaluate_config = evaluate_config
test_dataloader = DataLoader(
        test_ds, 
        batch_size = 1, 
        shuffle=False, 
        num_workers=2, 
        collate_fn=plate_collate_fn,
        drop_last=False
)




# load the model
ckpt = torch.load(train_config['load_ckpt'])
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

# we will relearn last layer, because i have fewer chars to be predicted
state_dict.pop('dense.weight', None)
state_dict.pop('dense.bias', None)

crnn = CRNN()
crnn.load_state_dict(state_dict, strict=False)

num_class = valid_ds.num_chars
crnn.to(device)

optimizer = optim.AdamW(crnn.parameters(), lr=train_config['lr']) # 

criterion = CTCLoss(reduction='sum', zero_infinity=True)
criterion.to(device)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50)



import math

def init_wandb():
        wandb.login(key='local-f57ccc8fdf497fe140ac4b5e383f9492de46ed93', host="http://localhost:8385")
        return wandb.init(project='license_plates', name='AdamW...')

logger = init_wandb()

logger.watch(crnn)

best_model = None
best_dist = float('inf')


import pandas as pd


for epoch in range(train_config['epochs']):
    tot_train_loss = 0.
    tot_train_count = 0
    crnn.train()

    # train per epoch
    with Timer(f'training step, epoch = {epoch}...'):
        for i, batch in enumerate(train_dataloader):
            images, targets, target_lengths = [d.to(device) for d in batch]

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)


            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
            target_lengths = torch.flatten(target_lengths)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(crnn.parameters(), 0.2) # gradient clipping with 5
            optimizer.step()

            train_size = batch[0].size(0)

            tot_train_loss += loss.item()
            tot_train_count += train_size

            wandb.log({'train_loss': loss.item() / train_size})
        
            if i % 50 == 0:
                print('train_batch_loss[', i, ']: ', loss.item() / train_size)
    scheduler.step()

    # validate after epoch
    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    lev_distance = 0
    wrong_cases = []
    crnn.eval()

    with Timer('validation step...'):
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                images, targets, target_lengths = [d.to(device) for d in batch]

                logits = crnn(images)
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)

                batch_size = images.size(0)
                input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
                target_lengths = torch.flatten(target_lengths)

                loss = criterion(log_probs, targets, input_lengths, target_lengths)

                preds = ctc_decode(log_probs, method=evaluate_config['decode_method'], beam_size=evaluate_config['beam_size'])
                reals = targets.cpu().numpy().tolist()
                target_lengths = target_lengths.cpu().numpy().tolist()

                tot_count += batch_size
                tot_loss += loss.item()
                target_length_counter = 0
                for pred, target_length in zip(preds, target_lengths):
                    real = reals[target_length_counter:target_length_counter + target_length]
                    target_length_counter += target_length

                    lev_distance += distance(pred, real)
                    if pred == real:
                        tot_correct += 1
                    # else:
                        #wrong_cases.append((real, pred))
    print(f'validation info: loss = {tot_loss / tot_count}; acc = {tot_correct / tot_count}; LEVENSTEIN = {lev_distance / tot_count}') # ; wrong_cases = {wrong_cases}'
    wandb.log({'valid_loss': tot_loss / tot_count, 'valid_acc': tot_correct / tot_count, 'valid_levenstein': lev_distance / tot_count})    
    
    
    if lev_distance / tot_count < best_dist:
        best_dist = lev_distance / tot_count
        best_model = copy.deepcopy(crnn)


checkpoint_path = train_config['checkpoints_dir'] + 'wtf.ckpt'
torch.save(best_model.state_dict(), checkpoint_path)
print('saved to', checkpoint_path)
print('Levenstein', best_dist)

# WANTED BASELINES: baseline20 = 0.05633 -> DONE; baseline25 = 0.03299 -> TODO


def get_pred(x):
    return ''.join([valid_ds.LABEL2CHAR[i] for i in x])


# validate after epoch
tot_count = 0
tot_loss = 0
tot_correct = 0
lev_distance = 0
wrong_cases = []

criterion = CTCLoss(reduction='sum', zero_infinity=True)
criterion.to(device)


crnn.eval()
with Timer('test step...'):
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            images, targets, target_lengths = [d.to(device) for d in batch]

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
            target_lengths = torch.flatten(target_lengths)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs, method=evaluate_config['decode_method'], beam_size=evaluate_config['beam_size'])
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length

                lev_distance += distance(pred, real)
                if pred == real:
                    tot_correct += 1
                    # else:
                        #wrong_cases.append((real, pred))

print(f'test info: loss = {tot_loss / tot_count}; acc = {tot_correct / tot_count}; LEVENSTEIN = {lev_distance / tot_count}') # ; wrong_cases = {wrong_cases}'
wandb.log({'test_loss': tot_loss / tot_count, 'test_acc': tot_correct / tot_count, 'test_levenstein': lev_distance / tot_count})    
wandb.finish()    






ckpt_path = checkpoint_path
state_dict = torch.load(ckpt_path, map_location="cpu")

crnn = CRNN()
crnn.load_state_dict(state_dict, strict=True)
crnn.eval()

crnn.to(device)

def get_pred(x):
    return ''.join([test_ds.LABEL2CHAR[i] for i in x])



all_ind = []
all_labels = [] 


all_ind = []
all_labels = []
with Timer('making predictions...'):
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            images, file_names = batch

            images = images.to(device)

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=evaluate_config['decode_method'], beam_size=evaluate_config['beam_size'])

            pred_labels = [get_pred(x) for x in preds]

            all_ind = all_ind + [x.split('/')[-1].split('.')[0] for x in file_names]
            all_labels = all_labels + pred_labels


df = pd.DataFrame(data = None)

df['index'] = all_ind
df['label'] = all_labels

df['index'] = df['index'].astype(int)

df.sort_values(by='index', inplace=True)

print(df.head())
df.to_csv('/mnt/calc/lizzzafomenko/rcnn/last_submission.csv', index = False)
