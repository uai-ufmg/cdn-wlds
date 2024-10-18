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

from data import WellLogDataset

from models import BYOL
from models import ResNet18, ResNet50, ResNet101, ResNet152
from models import dtw_distance

from utils import mean_reciprocal_rank, accuracy_at_k

from configs.config import ConfigArgs


def evaluate_similarity(model, test_dataloader, path_file, device):
    """
    Quantitatively evaluate the Similarity capacity of the BYOL and save the results.
        Arguments:
        ---------
            - model (nn.Module): Model object.
            - test_dataloader (torch.utils.data.DataLoader): Torch test dataloader used to pass data in batches.
            - path_file (str): Path to where the results will be saved.
            - device (str): Selected device to run the training code (CPU or GPU).
        Return:
        ---------
            None
    """
    
    latent_dataset = []

    print('Creating latent dataset')
    model.eval()
    model.net.train()
    model.online_encoder.eval()
    with torch.no_grad():
        for i, (wellname, sample_name, anchor_well, positive_well, positive_well2) in enumerate(tqdm(test_dataloader)):
            
            anchor_well = anchor_well.to(device)
            anchor_well = anchor_well.permute(0, 2, 1)
            
            latent_byol, _ = model(anchor_well, return_embedding=True)
            
            anchor_well = anchor_well.permute(0, 2, 1).cpu().detach().numpy()
            
            latent_dataset.append([wellname, sample_name, anchor_well, latent_byol])

    latent_dataset_vectors_byol = torch.stack([latent_dataset[i][3].squeeze(0) for i in range(len(latent_dataset))], dim=0)

    with torch.no_grad():
    
        y_pred_byol = []
        y_true = []
        sequences = []
        
        for i, (wellname, sample_name, anchor_well, positive_well, positive_well2) in enumerate(tqdm(test_dataloader)):

            sequences.append(sample_name[0])
            anchor_well = anchor_well.to(device)
            positive_well = positive_well.to(device)
            anchor_well = anchor_well.permute(0, 2, 1)
            positive_well = positive_well.permute(0, 2, 1)
            
            distances = []
    
            latent_byol_original, _ = model(anchor_well, return_embedding=True)
            latent_byol_augmented, _ = model(positive_well, return_embedding=True)
            
            positive_well = positive_well.permute(0, 2, 1).cpu().numpy()
    
            latent_dataset_vectors_byol_cpy = latent_dataset_vectors_byol.clone()
            latent_dataset_vectors_byol_cpy = torch.cat((latent_dataset_vectors_byol_cpy, latent_byol_augmented), dim=0)
            
            latent_dataset_cpy = copy.deepcopy(latent_dataset)
            latent_dataset_cpy.append(['augmentation', 'augmented', positive_well, latent_byol_augmented])
            
            for j in range(len(latent_dataset_cpy)):
                well2, pair_name, pair_sequence, latent_byol_pair = latent_dataset_cpy[j]
                if well2 == 'augmentation':
                    y_true.append(j)
                
            euclidean_byol_distance = F.pairwise_distance(latent_byol_original, latent_dataset_vectors_byol_cpy, keepdim = True).cpu().numpy()
            
            byol_distances = [[j, euclidean_byol_distance[j][0]] for j in range(len(euclidean_byol_distance)) if latent_dataset_cpy[j][0]!=wellname]
            
            anchor_well = anchor_well.cpu().numpy()
            byol_distances = sorted(byol_distances, reverse=False, key=lambda x: x[1])
    
            ranking_byol = [k[0] for k in byol_distances]
    
            y_pred_byol.append(ranking_byol)

        print(f'tamanho: {len(y_true)}')
        assert len(y_true) == len(y_pred_byol), "Y_true and Y_pred from BYOL should be the same length"

    metrics = dict()
    metrics['accuracy@1'] = dict()
    metrics['accuracy@5'] = dict()
    metrics['accuracy@10'] = dict()
    metrics['mrr'] = dict()

    metrics['accuracy@1']['BYOL'] = accuracy_at_k(y_true, y_pred_byol, k=1)
    metrics['accuracy@5']['BYOL'] = accuracy_at_k(y_true, y_pred_byol, k=5)    
    metrics['accuracy@10']['BYOL'] = accuracy_at_k(y_true, y_pred_byol, k=10)
    metrics['mrr']['BYOL'] = mean_reciprocal_rank(y_true, y_pred_byol)
    
    if os.path.isfile(path_file):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(path_file, write_mode) as f:
        f.write(f"Accuracy@1: {metrics['accuracy@1']['BYOL']}\nAccuracy@5: {metrics['accuracy@5']['BYOL']}\nAccuracy@10: {metrics['accuracy@10']['BYOL']}\nMRR: {metrics['mrr']['BYOL']}\n")


def train_byol(cfg, df_train, df_test, device):

    dataset = WellLogDataset(df_train, cfg['columns_used'], seq_size=cfg['seq_size'], interval_size=cfg['interval_size'])
    test_dataset = WellLogDataset(df_test, cfg['columns_used'], seq_size=cfg['seq_size'], interval_size=cfg['interval_size'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    backbone_byol = ResNet50(cfg['seq_size'], cfg['num_channels']).to(device)
    byol = BYOL(backbone_byol,
                image_size = (cfg['num_channels'], cfg['seq_size']),
                hidden_layer = -2,
                projection_size = cfg['seq_size'],
                projection_hidden_size = 4096,
                moving_average_decay = 0.99,
                use_momentum = True,
                sync_batchnorm = None)
    
    opt_byol = torch.optim.Adam(byol.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler_byol = torch.optim.lr_scheduler.StepLR(opt_byol, cfg['epochs'] // 10, 0.8)

    path_file = os.path.join(cfg['output_dir'], cfg['filename'])
    if not os.path.exists(cfg['output_dir']):
        raise FileNotFoundError(f"No such file or directory: {cfg['output_dir']}")

    inicio_treino = time.time()
    for epoch in range(cfg['epochs']):

        tic = time.time()
        
        byol.train()
        byol.net.train()
        byol.online_encoder.train()
        loss_list = []
        for i, (wellname, sample_name, anchor, positive1, positive2) in enumerate(dataloader):
            
            positive1 = positive1.float().to(device)
            positive1 = positive1.permute(0, 2, 1)
            
            positive2 = positive2.float().to(device)
            positive2 = positive2.permute(0, 2, 1)
    
            loss = byol.train_one_epoch(positive1, positive2, opt_byol)
    
            loss_list.append(loss.data.item())
        
        mean_loss = np.mean(loss_list)
        scheduler_byol.step()

        if epoch % 50== 0:
            evaluate_similarity(byol, test_dataloader, path_file, device)
        
        if (epoch+1) % 100 == 0:
            torch.save(byol.state_dict(), f'model-checkpoints/byol_{cfg["dataset"]}_{cfg["run"]}_epoch_{epoch+1}.pt')
            torch.save(byol.net.state_dict(), f'model-checkpoints/byol_backbone_{cfg["dataset"]}_{cfg["run"]}_epoch_{epoch+1}.pt')
            torch.save(byol.online_encoder.state_dict(), f'model-checkpoints/byol_online_encoder_{cfg["dataset"]}_{cfg["run"]}_epoch_{epoch+1}.pt')
            torch.save(byol.online_predictor.state_dict(), f'model-checkpoints/byol_online_predictor_{cfg["dataset"]}_{cfg["run"]}_epoch_{epoch+1}.pt')
    
        t = int(time.time() - tic)
        print(f'epoch: {epoch}/{cfg["epochs"]} - loss: {mean_loss:.4f} - {t // 60}m {t % 60}s', end='\r')
        
        if os.path.isfile(path_file):
            write_mode = "a"
        else:
            write_mode = "w"
        with open(path_file, write_mode) as f:
            f.write(f'Epoch: {epoch}/{cfg["epochs"]} - Loss: {mean_loss:.4f} - {t // 60}m {t % 60}s\n')

    final_treino = time.time()
    
    print(f'Tempo total de treinamento: {final_treino - inicio_treino}')
    if os.path.isfile(path_file):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(path_file, write_mode) as f:
        f.write(f'Tempo total de treinamento: {final_treino - inicio_treino}\n')
