import os
import time

import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data import WellLogDataset

from models import VAE
from models import dtw_distance

from utils import mean_reciprocal_rank, accuracy_at_k

from configs.config import ConfigArgs


def train(
    train_loader,
    train_set,
    model,
    opt_model,
    epoch,
    epochs,
    path_file,
    device):

    tic = time.time()

    # Setting networks for training mode.
    model.train()
    
    loss_list = []
    error_loss_list = []
    kl_loss_list = []

    # Iterating over batches.
    for i, (wellname, sample_name, anchor, positive1, positive2) in enumerate(train_loader):

        batch_size = anchor.shape[0]

        # Casting to correct device (x and z).
        anchor = anchor.to(device)
        anchor = anchor.permute(0, 2, 1)

        # Clearing the gradients of D optimizer.
        opt_model.zero_grad()

        # Forwarding data.
        reconstructed_sequence, mu, logvar = model(anchor)

        # Computing loss for data.
        loss = model.loss_function(anchor, reconstructed_sequence, mu, logvar, M_N = 0.0001)
        loss_value = loss['loss']
        kl_loss_value = loss['KLD']
        error_loss_value = loss['Reconstruction_Loss']

        # Computing backpropagation for model.
        loss_value.backward()

        # Taking step in model optimizer.
        opt_model.step()

        # Updating lists.
        loss_list.append(loss_value.data.item())
        error_loss_list.append(error_loss_value.data.item())
        kl_loss_list.append(kl_loss_value.data.item())
        
    # Printing training epoch loss.
    mean_loss = np.mean(loss_list)
    mean_error_loss = np.mean(error_loss_list)
    mean_kl_loss = np.mean(kl_loss_list)

    t = int(time.time() - tic)
    print(f'Epoch: {epoch}/{epochs} - Loss: {mean_loss:.4f} - Error Loss: {mean_error_loss:.4f} - KL Loss: {mean_kl_loss:.4f} - {t // 60}m {t % 60}s', end='\r')

    if os.path.isfile(path_file):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(path_file, write_mode) as f:
        f.write(f'Epoch: {epoch}/{epochs} - Loss: {mean_loss:.4f} - Error Loss: {mean_error_loss:.4f} - KL Loss: {mean_kl_loss:.4f} - {t // 60}m {t % 60}s\n')

    return loss_list, anchor, reconstructed_sequence


def evaluate(model, dataloader, columns_used, device):
    
    model.eval()
    #fig, ax = plt.subplots(3, num_channels, figsize=(20, 15))
    for i, (wellname, sample_name, anchor, positive1, positive2) in list(enumerate(dataloader))[:1]:
        
        anchor = anchor.to(device)
        anchor = anchor.permute(0, 2, 1)

        well_data_reconstructed, mu, logvar = model(anchor[:1])
        #well_data_reconstructed, latent_vector = model(anchor[:1])
        anchor = anchor.permute(0, 2, 1).cpu().detach().numpy()
        well_data_reconstructed = well_data_reconstructed.permute(0, 2, 1).cpu().detach().numpy()
        
        fig, axs = plt.subplots(len(columns_used), 1, figsize=(10, 8))  # 4 rows, 1 column
    
        for i in range(len(columns_used)):
            axs[i].plot(np.arange(0, len(anchor[0])), anchor[0][:,i])
            axs[i].plot(np.arange(0, len(well_data_reconstructed[0])), well_data_reconstructed[0][:,i], alpha=0.6)
            axs[i].set_title(f'{columns_used[i]}')
            axs[i].set_ylim(0,1)
            
        
        '''for j in range(num_channels):
            ax[i].invert_yaxis()
            ax[i].plot(np.arange(0, len(anchor[0])), anchor[0])
            ax[i].plot(np.arange(0, len(well_data_fake[0])), well_data_fake[0], alpha=0.6)'''
    plt.show()

def evaluate_similarity(model, test_dataloader, path_file, device):
    latent_dataset = []

    print('Creating latent dataset')
    model.eval()
    with torch.no_grad():
        for i, (wellname, sample_name, anchor_well, positive_well, positive_well2) in enumerate(tqdm(test_dataloader)):
            #print(idx)
            anchor_well = anchor_well.to(device)
            anchor_well = anchor_well.permute(0, 2, 1)
            
            latent_vae, _ = model.encode(anchor_well)
            
            anchor_well = anchor_well.permute(0, 2, 1).cpu().detach().numpy()
            
            latent_dataset.append([wellname, sample_name, anchor_well, latent_vae])

    latent_dataset_vectors_vae = torch.stack([latent_dataset[i][3].squeeze(0) for i in range(len(latent_dataset))], dim=0)

    with torch.no_grad():
    
        y_pred_vae = []
        y_pred_dtw = []
        y_true = []
        sequences = []
        
        for i, (wellname, sample_name, anchor_well, positive_well, positive_well2) in enumerate(tqdm(test_dataloader)):

            sequences.append(sample_name[0])
            anchor_well = anchor_well.to(device)
            positive_well = positive_well.to(device)
            anchor_well = anchor_well.permute(0, 2, 1)
            positive_well = positive_well.permute(0, 2, 1)
            
            distances = []
    
            latent_vae_original, _ = model.encode(anchor_well)
            latent_vae_augmented, _ = model.encode(positive_well)
            
            positive_well = positive_well.permute(0, 2, 1).cpu().numpy()
    
            latent_dataset_vectors_vae_cpy = latent_dataset_vectors_vae.clone()
            latent_dataset_vectors_vae_cpy = torch.cat((latent_dataset_vectors_vae_cpy, latent_vae_augmented), dim=0)
            
            latent_dataset_cpy = latent_dataset.copy()
            latent_dataset_cpy.append(['augmentation', 'augmented', positive_well, latent_vae_augmented])
            
            dtw_distances = []
            for j in range(len(latent_dataset_cpy)):
                well2, pair_name, pair_sequence, latent_vae_pair = latent_dataset_cpy[j]
                if well2 == 'augmentation':
                    y_true.append(j)
                #print(type(pair_sequence))
                #anchor_well_cpy = anchor_well.permute(0, 2, 1).cpu().numpy()
                pair_sequence_cpy = np.transpose(pair_sequence, (0, 2, 1))
                argument1 = anchor_well.squeeze(0).cpu()
                argument2 = torch.tensor(pair_sequence_cpy.squeeze(0)).cpu()
                dtw_distance_number = dtw_distance(argument1, argument2)
                if wellname != well2:
                    dtw_distances.append([j, dtw_distance_number])
                
            euclidean_vae_distance = F.pairwise_distance(latent_vae_original, latent_dataset_vectors_vae_cpy, keepdim = True).cpu().numpy()
            
            vae_distances = [[j, euclidean_vae_distance[j][0]] for j in range(len(euclidean_vae_distance)) if latent_dataset_cpy[j][0]!=wellname]
            
            anchor_well = anchor_well.cpu().numpy()
            vae_distances = sorted(vae_distances, reverse=False, key=lambda x: x[1])
            dtw_distances = sorted(dtw_distances, reverse=False, key=lambda x: x[1])
    
            ranking_dtw = [k[0] for k in dtw_distances]
            ranking_vae = [k[0] for k in vae_distances]
    
            y_pred_vae.append(ranking_vae)
            y_pred_dtw.append(ranking_dtw)
    
        assert len(y_true) == len(y_pred_vae), "Y_true and Y_pred from VAE should be the same length"
        assert len(y_true) == len(y_pred_dtw), "Y_true and Y_pred from DTW should be the same length"

    metrics = dict()
    metrics['accuracy@1'] = dict()
    metrics['accuracy@5'] = dict()
    metrics['accuracy@10'] = dict()
    metrics['mrr'] = dict()

    metrics['accuracy@1']['VAE'] = accuracy_at_k(y_true, y_pred_vae, k=1)
    metrics['accuracy@1']['DTW'] = accuracy_at_k(y_true, y_pred_dtw, k=1)
    
    metrics['accuracy@5']['VAE'] = accuracy_at_k(y_true, y_pred_vae, k=5)
    metrics['accuracy@5']['DTW'] = accuracy_at_k(y_true, y_pred_dtw, k=5)
    
    metrics['accuracy@10']['VAE'] = accuracy_at_k(y_true, y_pred_vae, k=10)
    metrics['accuracy@10']['DTW'] = accuracy_at_k(y_true, y_pred_dtw, k=10)
    
    metrics['mrr']['VAE'] = mean_reciprocal_rank(y_true, y_pred_vae)
    metrics['mrr']['DTW'] = mean_reciprocal_rank(y_true, y_pred_dtw)
    if os.path.isfile(path_file):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(path_file, write_mode) as f:
        f.write(f"Accuracy@1: {metrics['accuracy@1']['VAE']}\nAccuracy@5: {metrics['accuracy@5']['VAE']}\nAccuracy@10: {metrics['accuracy@10']['VAE']}\nMRR: {metrics['mrr']['VAE']}\n")
        f.write(f"Accuracy@1-DTW: {metrics['accuracy@1']['DTW']}\nAccuracy@5-DTW: {metrics['accuracy@5']['DTW']}\nAccuracy@10-DTW: {metrics['accuracy@10']['DTW']}\nMRR-DTW: {metrics['mrr']['DTW']}\n")


def train_vae(cfg, df_train, df_test, device):
    
    dataset = WellLogDataset(df_train, cfg['columns_used'], seq_size=cfg['seq_size'], interval_size=cfg['interval_size'])
    test_dataset = WellLogDataset(df_test, cfg['columns_used'], seq_size=cfg['seq_size'], interval_size=cfg['interval_size'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VAE(len(cfg['columns_used']), len(cfg['columns_used']), cfg['feature_size'], cfg['seq_size']).to(device)

    # Optimizer
    opt_model = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(opt_model, cfg['epochs'] // 5, 0.5)
    
    path_file = os.path.join(cfg['output_dir'], cfg['filename'])
    if not os.path.exists(cfg['output_dir']):
        raise FileNotFoundError(f"No such file or directory: {cfg['output_dir']}")

    # Lists for losses.
    train_loss = []
    
    inicio_treino = time.time()
    for epoch in range(cfg['epochs']):
        
        epc_loss, real_images, reconstructed_images = train(
            train_loader = dataloader,
            train_set = dataset,
            model = model,
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
            
        '''if epoch % 20 == 0:
            evaluate(model, dataloader, cfg['columns_used'], device)'''
        
        if (epoch+1) % 100 == 0:
            torch.save(model.state_dict(), f'model-checkpoints/autoencoder_{cfg["dataset"]}_{cfg["run"]}_epoch_{epoch+1}.pt')
    
    final_treino = time.time()
    
    print(f'Tempo total de treinamento: {final_treino - inicio_treino}')
    if os.path.isfile(path_file):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(path_file, write_mode) as f:
        f.write(f'Tempo total de treinamento: {final_treino - inicio_treino}\n')
