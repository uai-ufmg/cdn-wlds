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

from models import Discriminator, Encoder, Generator
from models import ResNet18, ResNet50, ResNet101, ResNet152

from utils import hybrid_z_sampling, signal_change_z_sampling
from utils import mean_reciprocal_rank, accuracy_at_k

from configs.config import ConfigArgs


def update_margin(epoch, initial_margin, final_margin, half_life=100):
    """
    Function responsible for giving updates to margin.
        Arguments:
        ---------
            - epoch (int): Number of the current epoch.
            - initial_margin (float): Value for the initial margin.
            - final_margin (float): Value for the final desired margin in the last epoch.
            - half_life (int): Number of epochs required to half the margin.
        Return:
        ---------
            - new_margin (float): Updated margin value to be used in the given epoch.
    """
    
    decay_rate = (-1/half_life) * np.log(1/2)
    new_margin = final_margin + (initial_margin-final_margin) * np.exp(-decay_rate * epoch)
    return new_margin


def train_epoch(
    train_loader,
    train_set,
    net_G, net_D, net_E, net_Sim,
    sim_loss, gan_loss,
    opt_GE, opt_D, opt_Sim,
    epoch, epochs, z_size,
    path_file, device):
    """
    Train one epoch of the model.
        Arguments:
        ---------
            - train_loader (torch.utils.data.DataLoader): Torch dataloader used to pass data in batches.
            - train_set (torch.utils.data.Dataset): Torch Dataset.
            - net_G (nn.Module): Generator.
            - net_D (nn.Module): Discriminator.
            - net_E (nn.Module): Encoder.
            - net_Sim (nn.Module): Similarity model.
            - sim_loss (nn.Module): Loss function to evaluate the similarity task (triplet loss).
            - gan_loss (nn.Module): Loss function to evaluate the generation task (cross-entropy).
            - opt_GE (torch.optim): Optimizer for both the Generator and Encoder.
            - opt_D (torch.optim): Optimizer for the discriminator.
            - opt_Sim (torch.optim): Optimizer for the similarity model.
            - epoch (int): Number of the current epoch.
            - epochs (int): Number of total epochs.
            - z_size (int): Size of the Generator's latent space
            - path_file (str): Path to where the results will be saved.
            - device (str): Selected device to run the training code (CPU or GPU).
        Return:
        ---------
            - loss_G_list (list): List of the Generator losses for all batches in the epoch.
            - loss_E_list (list): List of the Encoder losses for all batches in the epoch.
            - loss_D_list (list): List of the Discriminator losses for all batches in the epoch.
            - loss_Sim_list (list): List of the Similarity model losses for all batches in the epoch.
            - anchor (torch.Tensor): Last batch of anchors used for the Similarity model training.
            - G_out (torch.Tensor): Last batch of generated well log data.
    """

    tic = time.time()

    # Setting networks for training mode.
    net_D.train()
    net_G.train()
    net_E.train()
    net_Sim.train()
    
    loss_G_list = []
    loss_D_list = []
    loss_E_list = []
    loss_Sim_list = []

    # Iterating over batches.
    for i, (wellname, sample_name, anchor, positive1, positive2) in enumerate(train_loader):

        batch_size = anchor.shape[0]

        # Predefining ones and zeros for batches.
        ones_ = torch.ones(batch_size, 1).to(device)
        zeros_ = torch.zeros(batch_size, 1).to(device)

        # Casting to correct device (x and z).
        anchor = anchor.to(device)
        anchor = anchor.permute(0, 2, 1)
        positive1 = positive1.to(device)
        positive1 = positive1.permute(0, 2, 1)

        z = torch.randn(size=(batch_size, z_size, 1), device=device)
        
        #pair_first_depth = pair_first_depth.to(device)

        ##################
        # Updating net_D #
        ##################
        
        # Clearing the gradients of D optimizer.
        opt_D.zero_grad()

        real_encoded = net_E(anchor)

        # Forwarding real data.
        output_real = net_D(anchor, real_encoded.detach()) # Through D.

        # Computing loss for real data.
        D_real_loss = gan_loss(output_real, ones_)

        G_out = net_G(z) # Through G.
        output_fake = net_D(G_out.detach(), z) # Through D.

        # Computing loss for fake data.
        D_fake_loss = gan_loss(output_fake, zeros_)

        # Computing total loss for D.
        D_loss = (D_real_loss + D_fake_loss)/2

        # Computing backpropagation for D.
        D_loss.backward()

        # Taking step in D optimizer.
        opt_D.step()
        
        #####################
        # Updating G and E. #
        #####################
            
        # Clearing the gradients of GE optimizer.
        opt_GE.zero_grad()

        G_out = net_G(z) # Through G.
        real_encoded = net_E(anchor)

        output_fake = net_D(G_out, z) # Through D.
        output_real = net_D(anchor, real_encoded)

        # Computing loss for G and E.
        G_loss = gan_loss(output_fake, ones_)
        E_loss = gan_loss(output_real, zeros_)
        GE_loss = G_loss + E_loss

        # Computing backpropagation for G and E.
        GE_loss.backward()

        # Taking step in GE optimizer.
        opt_GE.step()

        ####################
        # Updating net_Sim #
        ####################
        
        # Clearing the gradients of D optimizer.
        opt_Sim.zero_grad()
        
        # Forwarding real data.
        output1_real, output2_real = net_Sim(anchor, positive1) # Through Sim.

        with torch.no_grad():
            z_encoded = net_E(anchor).detach()  # Ensure no gradients flow back
        # Creating random vector z.
        z = torch.randn(size=(batch_size, z_size, 1), device=device)
        z = hybrid_z_sampling(z, z_encoded, batch_size=batch_size, z_size=z_size, device=device)
        
        G_out = net_G(z) # Through G.
        output1_fake, output2_fake = net_Sim(anchor, G_out.detach()) # Through Sim.
        
        Sim_loss = sim_loss(output1_real, output2_real, output2_fake)

        # Computing backpropagation for D.
        Sim_loss.backward()

        # Taking step in D optimizer.
        opt_Sim.step()
        
        # Updating lists.
        loss_G_list.append(G_loss.data.item())
        loss_D_list.append(D_loss.data.item())
        loss_E_list.append(E_loss.data.item())
        loss_Sim_list.append(Sim_loss.data.item())
        
    # Printing training epoch loss.
    mean_loss_G = np.mean(loss_G_list)
    mean_loss_D = np.mean(loss_D_list)
    mean_loss_E = np.mean(loss_E_list)
    mean_loss_Sim = np.mean(loss_Sim_list)

    t = int(time.time() - tic)
    print(f'epoch: {epoch}/{epochs} - loss G: {mean_loss_G:.4f} - loss encoder: {mean_loss_E:.4f} - loss discrimination: {mean_loss_D:.4f} - loss similarity: {mean_loss_Sim:.4f} - {t // 60}m {t % 60}s', end='\r')

    if os.path.isfile(path_file):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(path_file, write_mode) as f:
        f.write(f'epoch: {epoch}/{epochs} - loss G: {mean_loss_G:.4f} - loss encoder: {mean_loss_E:.4f} - loss discrimination: {mean_loss_D:.4f} - loss similarity: {mean_loss_Sim:.4f} - margin: {sim_loss.margin:.2f} - {t // 60}m {t % 60}s\n')
    
    return loss_G_list, loss_E_list, loss_D_list, loss_Sim_list, anchor, G_out


def evaluate(net_G, net_E, dataloader, columns_used, z_size, device):
    """
    Visually evaluate the generation process of the GAN.
        Arguments:
        ---------
            - net_G (nn.Module): Generator.
            - net_E (nn.Module): Encoder.
            - dataloader (torch.utils.data.DataLoader): Torch dataloader used to pass data in batches.
            - columns_used (list[str]): List of selected logs.
            - z_size (int): Size of the Generator's latent space.
            - device (str): Selected device to run the training code (CPU or GPU).
        Return:
        ---------
            None
    """
    
    net_G.eval()
    net_E.eval()
    #fig, ax = plt.subplots(3, cfg.num_channels, figsize=(20, 15))
    for i, (wellname, sample_name, anchor, positive1, positive2) in list(enumerate(dataloader))[:1]:

        batch_size = anchor.shape[0]
        
        anchor = anchor.to(device)
        anchor = anchor.permute(0, 2, 1)

        with torch.no_grad():
            z_encoded = net_E(anchor).detach()  # Ensure no gradients flow back
        # Creating random vector z.
        z = torch.randn(size=(batch_size, z_size, 1), device=device)
        z = hybrid_z_sampling(z, z_encoded, batch_size=batch_size, z_size=z_size, device=device)
        
        well_data_fake = net_G(z)
        anchor = anchor.permute(0, 2, 1).cpu().detach().numpy()
        well_data_fake = well_data_fake.permute(0, 2, 1).cpu().detach().numpy()
        
        fig, axs = plt.subplots(len(columns_used), 1, figsize=(10, 8))  # 4 rows, 1 column
    
        for i in range(len(columns_used)):
            axs[i].plot(np.arange(0, len(anchor[0])), anchor[0][:,i])
            axs[i].plot(np.arange(0, len(well_data_fake[0])), well_data_fake[0][:,i], alpha=0.6)
            axs[i].set_title(f'{columns_used[i]}')
            axs[i].set_ylim(0,1)
            
    plt.show()


def evaluate_similarity(net_Sim, test_dataloader, path_file, device):
    """
    Quantitatively evaluate the Similarity capacity of the WellGT and save the results.
        Arguments:
        ---------
            - net_Sim (nn.Module): Similarity model.
            - test_dataloader (torch.utils.data.DataLoader): Torch test dataloader used to pass data in batches.
            - path_file (str): Path to where the results will be saved.
            - device (str): Selected device to run the training code (CPU or GPU).
        Return:
        ---------
            None
    """
    latent_dataset = []

    print('Creating latent dataset')
    net_Sim.eval()
    with torch.no_grad():
        for i, (wellname, sample_name, anchor_well, positive_well, positive_well2) in enumerate(tqdm(test_dataloader)):
            
            anchor_well = anchor_well.to(device)
            anchor_well = anchor_well.permute(0, 2, 1)
            
            latent_wellgt = net_Sim(anchor_well)
            
            anchor_well = anchor_well.permute(0, 2, 1).cpu().detach().numpy()
            
            latent_dataset.append([wellname, sample_name, anchor_well, latent_wellgt])

    latent_dataset_vectors_wellgt = torch.stack([latent_dataset[i][3].squeeze(0) for i in range(len(latent_dataset))], dim=0)

    with torch.no_grad():
    
        y_pred_wellgt = []
        y_true = []
        sequences = []
        
        for i, (wellname, sample_name, anchor_well, positive_well, positive_well2) in enumerate(tqdm(test_dataloader)):

            sequences.append(sample_name[0])
            anchor_well = anchor_well.to(device)
            positive_well = positive_well.to(device)
            anchor_well = anchor_well.permute(0, 2, 1)
            positive_well = positive_well.permute(0, 2, 1)
            
            distances = []
    
            latent_wellgt_original = net_Sim(anchor_well)
            latent_wellgt_augmented = net_Sim(positive_well)
            
            positive_well = positive_well.permute(0, 2, 1).cpu().numpy()
    
            latent_dataset_vectors_wellgt_cpy = latent_dataset_vectors_wellgt.clone()
            latent_dataset_vectors_wellgt_cpy = torch.cat((latent_dataset_vectors_wellgt_cpy, latent_wellgt_augmented), dim=0)
            
            latent_dataset_cpy = latent_dataset.copy()
            latent_dataset_cpy.append(['augmentation', 'augmented', positive_well, latent_wellgt_augmented])
            
            for j in range(len(latent_dataset_cpy)):
                well2, pair_name, pair_sequence, latent_wellgt_pair = latent_dataset_cpy[j]
                if well2 == 'augmentation':
                    y_true.append(j)
                
            euclidean_wellgt_distance = F.pairwise_distance(latent_wellgt_original, latent_dataset_vectors_wellgt_cpy, keepdim = True).cpu().numpy()
            
            wellgt_distances = [[j, euclidean_wellgt_distance[j][0]] for j in range(len(euclidean_wellgt_distance)) if latent_dataset_cpy[j][0]!=wellname]
            
            anchor_well = anchor_well.cpu().numpy()
            wellgt_distances = sorted(wellgt_distances, reverse=False, key=lambda x: x[1])
            
            ranking_wellgt = [k[0] for k in wellgt_distances]
    
            y_pred_wellgt.append(ranking_wellgt)
    
        assert len(y_true) == len(y_pred_wellgt), "Y_true and Y_pred from WellGT should be the same length"

    metrics = dict()
    metrics['accuracy@1'] = dict()
    metrics['accuracy@5'] = dict()
    metrics['accuracy@10'] = dict()
    metrics['mrr'] = dict()

    metrics['accuracy@1']['WellGT'] = accuracy_at_k(y_true, y_pred_wellgt, k=1)
    metrics['accuracy@5']['WellGT'] = accuracy_at_k(y_true, y_pred_wellgt, k=5)
    metrics['accuracy@10']['WellGT'] = accuracy_at_k(y_true, y_pred_wellgt, k=10)
    metrics['mrr']['WellGT'] = mean_reciprocal_rank(y_true, y_pred_wellgt)
    
    if os.path.isfile(path_file):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(path_file, write_mode) as f:
        f.write(f"Accuracy@1: {metrics['accuracy@1']['WellGT']}\nAccuracy@5: {metrics['accuracy@5']['WellGT']}\nAccuracy@10: {metrics['accuracy@10']['WellGT']}\nMRR: {metrics['mrr']['WellGT']}\n")


def train_wellgt(cfg, df_train, df_test, device):
    """
    Train the model for n epochs and save the results.
        Arguments:
        ---------
            - cfg (dict): Config dictionary that contains all info loaded right after running the script.
            - df_train (pd.DataFrame): Train dataframe containing all logs and info needed.
            - df_test (pd.DataFrame): Test dataframe containing all logs and info needed.
            - device (str): Selected device to run the training code (CPU or GPU).
        Return:
        ---------
            None
    """

    dataset = WellLogDataset(df_train, cfg['columns_used'], seq_size=cfg['seq_size'], interval_size=cfg['interval_size'])
    test_dataset = WellLogDataset(df_test, cfg['columns_used'], seq_size=cfg['seq_size'], interval_size=cfg['interval_size'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # MODELS
    net_G = Generator(cfg['z_size'], len(cfg['columns_used'])).to(device=device)
    net_D = Discriminator(len(cfg['columns_used']), cfg['z_size']).to(device=device)
    net_E = Encoder(len(cfg['columns_used']), cfg['z_size']).to(device=device)
    net_Sim = ResNet50(cfg['feature_size'], len(cfg['columns_used'])).to(device=device)

    # LOSSES
    sim_loss = nn.TripletMarginLoss(margin=cfg['initial_margin'], p=2, eps=1e-7, swap=cfg['swap'])
    gan_loss = nn.BCELoss()
    
    # OPTIMIZERS
    opt_GE = torch.optim.Adam(list(net_G.parameters()) + list(net_E.parameters()), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    opt_D = torch.optim.Adam(net_D.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    opt_Sim = torch.optim.Adam(net_Sim.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    
    # SCHEDULERS
    scheduler_GE = torch.optim.lr_scheduler.StepLR(opt_GE, cfg['epochs'] // 10, 0.95)
    scheduler_D = torch.optim.lr_scheduler.StepLR(opt_D, cfg['epochs'] // 10, 0.95)
    scheduler_Sim = torch.optim.lr_scheduler.StepLR(opt_Sim, cfg['epochs'] // 10, 0.95)

    path_file = os.path.join(cfg['output_dir'], cfg['filename'])
    if not os.path.exists(cfg['output_dir']):
        raise FileNotFoundError(f"No such file or directory: {cfg['output_dir']}")
    
    # Lists for losses.
    train_loss_G = []
    train_loss_D = []
    train_loss_E = []
    train_loss_Sim = []
    
    test_fake_images = []
    test_real_images = []
    similarity_result_test = []
    
    inicio_treino = time.time()
    for epoch in range(cfg['epochs']):
        epc_loss_G, epc_loss_E, epc_loss_D, epc_loss_Sim, real_images, fake_images = train_epoch(
            train_loader = dataloader,
            train_set = dataset,
            net_G = net_G,
            net_D = net_D,
            net_E = net_E,
            net_Sim = net_Sim,
            sim_loss = sim_loss,
            gan_loss = gan_loss,
            opt_GE = opt_GE,
            opt_D = opt_D,
            opt_Sim = opt_Sim,
            epoch = epoch,
            epochs = cfg['epochs'],
            z_size = cfg['z_size'],
            path_file = path_file,
            device = device
        )

        current_margin = update_margin(epoch, initial_margin=cfg['initial_margin'], final_margin=cfg['final_margin'],  half_life=cfg['half_life'])
        sim_loss.margin = current_margin
        
        train_loss_G = train_loss_G + epc_loss_G
        train_loss_D = train_loss_D + epc_loss_D
        train_loss_E = train_loss_E + epc_loss_E
        train_loss_Sim = train_loss_Sim + epc_loss_Sim
        
        mean_epc_loss_G = np.mean(epc_loss_G)
        mean_epc_loss_D = np.mean(epc_loss_D)
        mean_epc_loss_E = np.mean(epc_loss_E)
        mean_epc_loss_Sim = np.mean(epc_loss_Sim)
    
        # Taking step on scheduler.
        scheduler_GE.step()
        scheduler_D.step()
        scheduler_Sim.step()
        
        if epoch % 50== 0:
        #    evaluate(net_G, net_E, dataloader, cfg['columns_used'], cfg['z_size'], device)
            evaluate_similarity(net_Sim, test_dataloader, path_file, device)

        if cfg['save_model'] and ((epoch+1) % 100 == 0):
            torch.save(net_G.state_dict(), os.path.join(cfg['save_dir'], f'generator_epoch_{epoch+1}.pt'))
            torch.save(net_D.state_dict(), os.path.join(cfg['save_dir'], f'discriminator_epoch_{epoch+1}.pt'))
            torch.save(net_E.state_dict(), os.path.join(cfg['save_dir'], f'encoder_epoch_{epoch+1}.pt'))
            torch.save(net_Sim.state_dict(), os.path.join(cfg['save_dir'], f'similarity_model_epoch_{cfg["run"]}_{epoch+1}.pt'))
    
    final_treino = time.time()
    
    print(f'Tempo total de treinamento: {final_treino - inicio_treino}')
    if os.path.isfile(path_file):
        write_mode = "a"
    else:
        write_mode = "w"
    with open(path_file, write_mode) as f:
        f.write(f'Tempo total de treinamento: {final_treino - inicio_treino}\n')

