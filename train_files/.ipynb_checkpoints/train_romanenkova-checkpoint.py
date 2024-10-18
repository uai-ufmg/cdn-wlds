import os
import time
import copy

import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data import RomanenkovaDataset, WellLogDataset

from models import ResNet18, ResNet50

from utils import mean_reciprocal_rank, accuracy_at_k

from configs.config import ConfigArgs


def train(
    train_loader,
    train_set,
    model,
    criterion,
    opt_model,
    epoch,
    epochs,
    path_file,
    device):

    tic = time.time()

    # Setting networks for training mode.
    model.train()
    
    loss_list = []

    # Iterating over batches.
    for i, (wellname, sample_name, anchor_well, positive_well, negative_well) in enumerate(train_loader):

        batch_size = anchor_well.shape[0]

        # Casting to correct device (x and z).
        anchor_well = anchor_well.to(device)
        anchor_well = anchor_well.permute(0, 2, 1)
        
        positive_well = positive_well.to(device)
        positive_well = positive_well.permute(0, 2, 1)
        
        negative_well = negative_well.to(device)
        negative_well = negative_well.permute(0, 2, 1)

        # Clearing the gradients of D optimizer.
        opt_model.zero_grad()

        # Forwarding data.
        anchor_representation = model.forward_once(anchor_well) # Anchor
        positive_representation = model.forward_once(positive_well) # Positive
        negative_representation = model.forward_once(negative_well) # Negative

        # Computing loss for data.
        loss = criterion(anchor_representation, positive_representation, negative_representation)

        # Computing backpropagation for model.
        loss.backward()

        # Taking step in model optimizer.
        opt_model.step()

        # Updating lists.
        loss_list.append(loss.data.item())
        
    # Printing training epoch loss.
    mean_loss = np.mean(loss_list)

    t = int(time.time() - tic)
    print(f'Epoch: {epoch}/{epochs} - Loss: {mean_loss:.4f} - {t // 60}m {t % 60}s', end='\r')
    
    if os.path.isfile(path_file):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(path_file, write_mode) as f:
        f.write(f'Epoch: {epoch}/{epochs} - Loss: {mean_loss:.4f} - {t // 60}m {t % 60}s\n')

    return loss_list


def evaluate_similarity(model, test_dataloader, path_file, device):
    latent_dataset = []

    print('Creating latent dataset')
    model.eval()
    with torch.no_grad():
        for i, (wellname, sample_name, anchor_well, positive_well, positive_well2) in enumerate(tqdm(test_dataloader)):
            #print(idx)
            anchor_well = anchor_well.to(device)
            anchor_well = anchor_well.permute(0, 2, 1)
            
            latent_romanenkova = model(anchor_well)
            
            anchor_well = anchor_well.permute(0, 2, 1).cpu().detach().numpy()
            
            latent_dataset.append([wellname, sample_name, anchor_well, latent_romanenkova])

    latent_dataset_vectors_romanenkova = torch.stack([latent_dataset[i][3].squeeze(0) for i in range(len(latent_dataset))], dim=0)

    with torch.no_grad():
    
        y_pred_romanenkova = []
        y_true = []
        sequences = []
        
        for i, (wellname, sample_name, anchor_well, positive_well, positive_well2) in enumerate(tqdm(test_dataloader)):

            sequences.append(sample_name[0])
            anchor_well = anchor_well.to(device)
            positive_well = positive_well.to(device)
            anchor_well = anchor_well.permute(0, 2, 1)
            positive_well = positive_well.permute(0, 2, 1)
            
            distances = []
    
            latent_romanenkova_original = model(anchor_well)
            latent_romanenkova_augmented = model(positive_well)
            
            positive_well = positive_well.permute(0, 2, 1).cpu().numpy()
    
            latent_dataset_vectors_romanenkova_cpy = latent_dataset_vectors_romanenkova.clone()
            latent_dataset_vectors_romanenkova_cpy = torch.cat((latent_dataset_vectors_romanenkova_cpy, latent_romanenkova_augmented), dim=0)
            
            latent_dataset_cpy = latent_dataset.copy()
            latent_dataset_cpy.append(['augmentation', 'augmented', positive_well, latent_romanenkova_augmented])
            
            for j in range(len(latent_dataset_cpy)):
                well2, pair_name, pair_sequence, latent_romanenkova_pair = latent_dataset_cpy[j]
                if well2 == 'augmentation':
                    y_true.append(j)
                
            euclidean_romanenkova_distance = F.pairwise_distance(latent_romanenkova_original, latent_dataset_vectors_romanenkova_cpy, keepdim = True).cpu().numpy()
            
            romanenkova_distances = [[j, euclidean_romanenkova_distance[j][0]] for j in range(len(euclidean_romanenkova_distance)) if latent_dataset_cpy[j][0]!=wellname]
            
            anchor_well = anchor_well.cpu().numpy()
            romanenkova_distances = sorted(romanenkova_distances, reverse=False, key=lambda x: x[1])
    
            ranking_romanenkova = [k[0] for k in romanenkova_distances]
    
            y_pred_romanenkova.append(ranking_romanenkova)

        print(f'tamanho: {len(y_true)}')
        assert len(y_true) == len(y_pred_romanenkova), "Y_true and Y_pred from Romanenkova should be the same length"

    metrics = dict()
    metrics['accuracy@1'] = dict()
    metrics['accuracy@5'] = dict()
    metrics['accuracy@10'] = dict()
    metrics['mrr'] = dict()

    metrics['accuracy@1']['Romanenkova'] = accuracy_at_k(y_true, y_pred_romanenkova, k=1)
    
    metrics['accuracy@5']['Romanenkova'] = accuracy_at_k(y_true, y_pred_romanenkova, k=5)
    
    metrics['accuracy@10']['Romanenkova'] = accuracy_at_k(y_true, y_pred_romanenkova, k=10)
    
    metrics['mrr']['Romanenkova'] = mean_reciprocal_rank(y_true, y_pred_romanenkova)
    if os.path.isfile(path_file):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(path_file, write_mode) as f:
        f.write(f"Accuracy@1: {metrics['accuracy@1']['Romanenkova']}\nAccuracy@5: {metrics['accuracy@5']['Romanenkova']}\nAccuracy@10: {metrics['accuracy@10']['Romanenkova']}\nMRR: {metrics['mrr']['Romanenkova']}\n")


def train_romanenkova(cfg, df_train, df_test, device):
    
    dataset = RomanenkovaDataset(df_train, cfg['columns_used'], seq_size=cfg['seq_size'], interval_size=cfg['interval_size'])
    test_dataset = WellLogDataset(df_test, cfg['columns_used'], seq_size=cfg['seq_size'], interval_size=cfg['interval_size'])
    
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = ResNet50(cfg['feature_size'], len(cfg['columns_used'])).to(device)

    sim_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7, swap=True)

    # Optimizer
    opt_model = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(opt_model, cfg['epochs'] // 10, 0.95)

    path_file = os.path.join(cfg['output_dir'], cfg['filename'])
    if not os.path.exists(cfg['output_dir']):
        raise FileNotFoundError(f"No such file or directory: {cfg['output_dir']}")
    
    # Lists for losses.
    train_loss = []

    inicio_treino = time.time()
    for epoch in range(cfg['epochs']):
        
        epc_loss = train(
            train_loader = dataloader,
            train_set = dataset,
            model = model,
            criterion = sim_loss,
            opt_model = opt_model,
            epoch = epoch,
            epochs = cfg['epochs'],
            path_file = path_file,
            device = device
        )
    
        train_loss = train_loss + epc_loss
        
        mean_epc_loss = np.mean(epc_loss)
    
        # Taking step on scheduler.
        scheduler.step()
        
        if epoch % 50 == 0:
            evaluate_similarity(model, test_dataloader, path_file, device)
        
        model.eval()

        if cfg['save_model'] and ((epoch+1) % 100 == 0):
            torch.save(model.state_dict(), os.path.join(cfg['save_dir'], f'romanenkova_epoch_{epoch+1}.pt'))
    
    final_treino = time.time()
    
    print(f'Tempo total de treinamento: {final_treino - inicio_treino}')
    if os.path.isfile(path_file):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(path_file, write_mode) as f:
        f.write(f'Tempo total de treinamento: {final_treino - inicio_treino}\n')
