import os
from pathlib import Path
import argparse
import yaml
import numpy as np


class ConfigArgs():
    
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='wellgt', help='Model name')
        parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
        parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight Decay')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
        parser.add_argument('--epochs', type=int, default=500, help='Epochs')

        parser.add_argument('--feature_size', type=int, default=256, help='Feature Size')
        parser.add_argument('--z_size', type=int, default=100, help='Z vector size (for the GAN generator)')

        parser.add_argument('--dataset', type=str, default='public', help='Dataset name')
        parser.add_argument('--seq_size', type=int, default=256, help='Sequence size')
        parser.add_argument('--split_method', type=str, default='tt', help='Method used to split data')
        parser.add_argument('--interval_size', type=int, default=128, help='Interval between two sequences')
        parser.add_argument('--initial_margin', type=float, default=0.1, help='Initial Triplet Loss Margin')
        parser.add_argument('--final_margin', type=float, default=0.1, help='Final Triplet Loss Margin')
        parser.add_argument('--half_life', type=int, default=100, help='Half Life Triplet Loss Margin')
        parser.add_argument('--swap', type=self.__str2bool, default=True, help='Triplet Loss Swap')
        
        parser.add_argument('--run', type=int, default=1, help='Execution number')
        
        parser.add_argument("--save_model", type=self.__str2bool, nargs='?', const=True, default=True, help="Save trained model")
        parser.add_argument("--save_dir", type=Path, default=Path("model-checkpoints"), help="Path to save model weights")
        
        parser.add_argument("--output_dir", type=Path, default=Path("results"), help="Path to save logs")
        parser.add_argument("--config_dir", type=Path, default=Path("configs"), help="Path to load config files")
        parser.add_argument("--verbose", type=self.__str2bool, nargs='?', const=True, default=False, help="Use verbose")
        
        self.parser = parser
    
    def __str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    def parse_args(self):
        
        args = self.parser.parse_args()

        cfg = dict()

        cfg['learning_rate'] = args.lr
        cfg['weight_decay'] = args.weight_decay
        cfg['batch_size'] = args.batch_size
        cfg['epochs'] = args.epochs

        cfg['feature_size'] = args.feature_size
        cfg['z_size'] = args.z_size
        
        cfg['seq_size'] = args.seq_size
        cfg['split_method'] = args.split_method
        cfg['interval_size'] = args.interval_size

        cfg['initial_margin'] = args.initial_margin
        cfg['final_margin'] = args.final_margin
        cfg['half_life'] = args.half_life
        cfg['swap'] = args.swap

        cfg['config_dir'] = args.config_dir
        cfg['save_model'] = args.save_model
        cfg['save_dir'] = args.save_dir
        cfg['output_dir'] = args.output_dir
        
        cfg['verbose'] = args.verbose
        cfg['run'] = args.run
        
        model_name = args.model.lower()
        if model_name == 'wellgt':
            model_name = 'WellGT'
        elif model_name == 'romanenkova':
            model_name = 'Romanenkova'
        elif model_name == 'byol':
            model_name = 'BYOL'
        elif model_name == 'vae':
            model_name = 'VAE'
        else:
            raise NotImplementedError('Model name does not exist')

        cfg['model_name'] = model_name
        cfg['data_dir'] = []
        cfg['dataset'] = args.dataset.lower()
        
        if cfg['dataset']=='public':
            with open(os.path.join(cfg['config_dir'], f'taranaki.yml'), 'r') as file:
                cfg_data = yaml.safe_load(file)
                cfg['data_dir'].append(cfg_data['data_dir'])
            with open(os.path.join(cfg['config_dir'], f'force.yml'), 'r') as file:
                cfg_data = yaml.safe_load(file)
                cfg['data_dir'].append(cfg_data['data_dir'])
            cfg['dataset_names'] = ['taranaki', 'force']
        else:
            with open(os.path.join(cfg['config_dir'], f'{cfg["dataset"]}.yml'), 'r') as file:
                cfg_data = yaml.safe_load(file)
                cfg['data_dir'].append(cfg_data['data_dir'])
            cfg['dataset_names'] = [cfg['dataset']]
            
        with open(os.path.join(cfg['config_dir'], f'data.yml'), 'r') as file:
            cfg_data = yaml.safe_load(file)
            cfg = {**cfg,**cfg_data}

        cfg['filename'] = f"{cfg['model_name']}_{cfg['dataset']}_{cfg['seq_size']}_{cfg['interval_size']}_{cfg['run']}"
        
        return cfg
