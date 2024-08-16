import torch
import sys
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from datetime import datetime
import wandb
import random
import copy
import torch.nn.functional as F
import json
import numpy as np
from sklearn.model_selection import KFold
import math

from utils import get_patient_by_id_standardized, get_patient_by_id_original
from config import *
from loss import UtilityLoss

def get_dataloaders(data, train_idx, val_idx, batch_size=256, num_workers=24):
    train_subset = Subset(data, train_idx)
    val_subset = Subset(data, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
    

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def make_layers(self, layers_config):
        layers = []

        for layer in layers_config:
            layer_type = layer.pop('type')
            if layer_type == 'RNNLayer':
                layer_class = RNNModel.RNNLayer
            elif layer_type == 'GRULayer':
                layer_class = GRUModel.GRUCell
            else:
                layer_class = globals().get(layer_type)
                if not layer_class:
                    layer_class = getattr(nn, layer_type)

            layers.append(layer_class(**layer))
        return layers

    def load_saved_model(self):
        if self.model_path:
            state_dict = torch.load(self.model_path, map_location='cpu')
            cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            try:
                self.load_state_dict(cleaned_state_dict)
                print('Loaded saved model ' + self.model_path)
            except:
                print('Failed loading model ' + self.model_path)
                sys.exit()
        else:
            print('Not using saved model.')
            
        return

    def save_model(self, model, rid, epoch, loss):
        if not os.path.exists('../models'):
            os.mkdir('../models')
            # os.mkdir(dirpath)
        now = datetime.now()
        timestr = now.strftime("%m_%d_%Y_%H_%M_%S")
        model_path = '../models/{}_{}_{:.5f}_{}.pth'.format(rid, epoch, loss, timestr)
        torch.save(copy.deepcopy(model).cpu().state_dict(), model_path)
        self.model_path = model_path
        return

    def forward(self, x, **kwargs):
        return self.model(x)
    
    def train_model(self, dataset, use_val=False, epochs=50, batch_size=256, pos_weight=54.5, lr=0.001, loss_criterion='BCE', logging=False, num_workers=24):
        if use_val:
            data_indices = np.arange(len(dataset))
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            folds = []
            for train_idx, val_idx in kf.split(data_indices):
                folds.append((train_idx, val_idx))
            num_folds = len(folds)
            train_len = len(folds[0][0])
            valid_len = len(folds[0][1])
            assert train_len + valid_len == len(dataset), "Error during train-validation split!"
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            valid_loader = None
            train_len = len(dataset)

        method = self.method
        
        # start a new wandb run to track this script
        if logging:
            run = wandb.init(
                project = wandb_project,
                config = {
                    "architecture"       : method,
                    "model_config"       : self.config,
                    "dataset"            : "Competition2019",
                    "preprocessing"      : "standardized",
                    "batch_size"         : batch_size,
                    "learning_rate"      : lr,
                    "epochs"             : epochs,
                    "pos_weight"         : pos_weight
                }
            )
            rid = method + '_' + run.name
        else:
            rid = method + '_trial'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        model = self.setup_model(device)
        # model = self.model
        # model = model.to(device)
        # model = torch.nn.DataParallel(model)

        if loss_criterion == 'BCE':
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])).to(device)
        elif loss_criterion == 'Utility':
            criterion = UtilityLoss(pos_weight).to(device)
        else:
            print('Error with Loss Criterion {}'.format(loss_criterion))
            sys.exit()

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

        try:
            for epoch in range(epochs):
                print('====== Epoch {} ======'.format(epoch))
                if use_val:
                    current_fold = epoch % num_folds
                    train_idx, val_idx = folds[current_fold]
                    train_loader, valid_loader = get_dataloaders(dataset, train_idx, val_idx, batch_size, num_workers)

                model.train()
                total_loss = 0

                for batch in tqdm(train_loader):
                    _, _, x_batch, y_batch, u_batch, mask_batch, _ = batch
                    y_batch = y_batch.unsqueeze(1)
                    if self.method == 'ResNet':
                        x_batch = x_batch.unsqueeze(1)
                    x_batch, y_batch, u_batch = x_batch.to(device), y_batch.to(device), u_batch.to(device)
                    optimizer.zero_grad()
                    if self.method == 'Transformer':
                        mask_batch = mask_batch.to(device)
                        outputs = self.forward(x_batch, mask=mask_batch)
                    else:
                        outputs = model(x_batch)
                    if not loss_criterion == 'Utility':
                        loss = criterion(outputs, y_batch)
                    else:
                        loss = criterion(outputs, y_batch, u_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    total_loss += loss.item()
                print(f'Train Loss: {total_loss / train_len}')

                if valid_loader:
                    # Validation loop
                    model.eval()
                    total_val_loss = 0
                    with torch.no_grad():
                        for batch in valid_loader:
                            _, _, x_batch, y_batch, u_batch, mask_batch, _ = batch
                            y_batch = y_batch.unsqueeze(1)
                            if self.method == 'ResNet':
                                x_batch = x_batch.unsqueeze(1)
                            x_batch, y_batch, u_batch = x_batch.to(device), y_batch.to(device), u_batch.to(device)
                            if self.method == 'Transformer':
                                mask_batch = mask_batch.to(device)
                                outputs = self.forward(x_batch, mask=mask_batch)
                            else:
                                outputs = model(x_batch)
                            if not loss_criterion == 'Utility':
                                loss = criterion(outputs, y_batch)
                            else:
                                loss = criterion(outputs, y_batch, u_batch)
                            total_val_loss += loss.item()
                    print(f'Validation Loss: {total_val_loss / valid_len}')

                if logging:
                    if valid_loader:
                        wandb.log({
                            "Train loss"     : total_loss     / train_len,
                            "Validation loss": total_val_loss / valid_len
                        })
                    else:
                        wandb.log({
                            "Train loss"     : total_loss     / train_len
                        })

                if (epoch+1) % 5 == 0:
                    epoch_loss = total_loss / train_len
                    self.save_model(model, rid, epoch, epoch_loss)

            self.load_saved_model()

        except KeyboardInterrupt:
            print("\nTraining interrupted by the user at Epoch {}.".format(epoch))
            print("Last saved {}".format(5*((epoch+1)//5)-1))

        except Exception as e:
            print("An error occurred:", str(e))
            print("Last saved {}".format(5*((epoch+1)//5)-1))
            pass

        if logging:
            wandb.finish()
        
        return rid
    
    def setup_model(self, device):
        if hasattr(self, 'model'):
            model = self.model
            model = model.to(device)
            model = torch.nn.DataParallel(model)
        else:
            model = self.to(device)
            model = torch.nn.DataParallel(model)
        return model
    
    
class LogisticRegressionModel(BaseModel):
    def __init__(self, input_size, output_size, config, model_path=None):
        super().__init__()
        self.method = 'Log'
        self.config = config
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size[0]*input_size[1], output_size),
            nn.Sigmoid()
        )
        self.model_path = model_path
        self.load_saved_model()
        

            
class MLPModel(BaseModel):
    def __init__(self, input_size, output_size, config, model_path=None):
        super().__init__()
        self.method = 'MLP'
        self.config = config
        self.layers = self.make_layers(config['model']['layers'])
        self.model = nn.Sequential(*self.layers)
        self.model_path = model_path
        self.load_saved_model()

# Recurrent Neural Network
class RNNModel(BaseModel):
    class RNNLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.input_to_hidden = nn.Linear(in_features, out_features)
            self.hidden_to_hidden = nn.Linear(out_features, out_features)
            self.out_features = out_features

        def forward(self, x, h):
            h = torch.tanh(self.input_to_hidden(x) + self.hidden_to_hidden(h))
            return h

    def __init__(self, config, model_path=None):
        super().__init__()
        self.method = 'RNN'
        self.config = config
        self.input_shape = config['model']['input_shape'][1]
        self.hidden_size = config['model']['hidden_size']
        self.output_size = config['model']['output_size']
        self.num_layers = config['model']['num_layers']

        # Initial layer
        self.initial_layer = nn.Linear(self.input_shape, self.hidden_size)

        # Recurrent layers
        self.recurrent_layers = nn.ModuleList([
            self.RNNLayer(self.hidden_size, self.hidden_size)
            for _ in range(self.num_layers)
        ])

        # Final output layer
        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )

        # # Print recurrent layer parameters
        # print("recurrent_layers parameters:")
        # for param in self.recurrent_layers.parameters():
        #     print(param)

        self.model_path = model_path
        self.init_weights()
        self.load_saved_model()

    def init_weights(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        batch_size, seq_len, feature_size = x.size()
        
        # Pass through initial layer
        x = self.initial_layer(x)
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        # Process each time step
        for t in range(seq_len):
            for layer in self.recurrent_layers:
                h = layer(x[:, t, :], h)
        
        # Pass through final layer
        y = self.final_layer(h)
        return y

    def parameters(self):
        return list(self.initial_layer.parameters()) + list(self.recurrent_layers.parameters()) + list(self.final_layer.parameters())

    def train_model(self, dataset, use_val=False, epochs=50, batch_size=256, pos_weight=54.5, lr=0.001, loss_criterion='BCE', logging=False, num_workers=24):
        rid = super().train_model(
            dataset=dataset,
            use_val=use_val,
            epochs=epochs,
            batch_size=batch_size,
            pos_weight=pos_weight,
            lr=lr,
            loss_criterion=loss_criterion,
            logging=logging,
            num_workers=num_workers
        )
        
        return rid
    
    def setup_model(self, device):
        self.to(device)
        self.initial_layer.to(device)
        self.recurrent_layers.to(device)
        self.final_layer.to(device)
        
        return self

class LSTMModel(BaseModel):
    class LSTMCell(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size

            # Gates
            self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
            self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
            self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
            self.candidate_gate = nn.Linear(input_size + hidden_size, hidden_size)

        def forward(self, x, h_prev, c_prev):
            # Concatenate input and hidden state
            combined = torch.cat((x, h_prev), dim=1)

            # Forget gate
            f_t = torch.sigmoid(self.forget_gate(combined))
            
            # Input gate
            i_t = torch.sigmoid(self.input_gate(combined))
            
            # Candidate memory cell
            c_hat_t = torch.tanh(self.candidate_gate(combined))
            
            # Final memory cell
            c_t = f_t * c_prev + i_t * c_hat_t
            
            # Output gate
            o_t = torch.sigmoid(self.output_gate(combined))
            
            # Final hidden state
            h_t = o_t * torch.tanh(c_t)

            return h_t, c_t
    
    def __init__(self, config, model_path=None):
        super().__init__()
        self.method = 'LSTM'
        self.config = config
        
        # LSTM Configuration
        self.input_size = config['model']['input_shape'][1]  # Number of features
        self.hidden_size = config['model']['hidden_size']    # Hidden state size
        self.num_layers = config['model']['num_layers']      # Number of LSTM layers
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList([self.LSTMCell(self.input_size if i == 0 else self.hidden_size, self.hidden_size) for i in range(self.num_layers)])
        
        # Final output layer
        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_size, config['model']['output_size']),
            nn.Sigmoid()
        )


        self.model_path = model_path
        self.load_saved_model()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Initialize hidden and cell states for each layer
        h_t = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        
        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_t[layer], c_t[layer] = self.lstm_layers[layer](x_t, h_t[layer], c_t[layer])
                x_t = h_t[layer]  # Input to the next layer is the output of the current layer
        
        # Pass through the linear and ReLU layers
        x = self.fc1(h_t[-1])  # Use the output of the last LSTM layer
        x = self.relu(x)
        y = self.fc2(x)
        
        return y
    
    def setup_model(self, device):
        self.to(device)
        return self

class GRUModel(BaseModel):
    class GRUCell(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(GRUModel.GRUCell, self).__init__()
            self.hidden_dim = hidden_dim

            # Update gate parameters
            self.Wz = nn.Linear(input_dim, hidden_dim)
            self.Uz = nn.Linear(hidden_dim, hidden_dim)

            # Reset gate parameters
            self.Wr = nn.Linear(input_dim, hidden_dim)
            self.Ur = nn.Linear(hidden_dim, hidden_dim)

            # New memory content parameters
            self.Wh = nn.Linear(input_dim, hidden_dim)
            self.Uh = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x, h_prev):
            z = torch.sigmoid(self.Wz(x) + self.Uz(h_prev))
            r = torch.sigmoid(self.Wr(x) + self.Ur(h_prev))
            h_hat = torch.tanh(self.Wh(x) + self.Uh(r * h_prev))
            h = (1 - z) * h_prev + z * h_hat
            return h

    def __init__(self, config, model_path=None):
        super().__init__()
        self.method = 'GRU'
        self.config = config
        self.input_shape = config['model']['input_shape']
        self.hidden_size = config['model']['hidden_size']
        self.output_size = config['model']['output_size']
        self.num_layers = config['model']['num_layers']

        # Initialize GRU layers and output layer
        self.initial_layer = nn.Linear(self.input_shape[1], self.hidden_size)
        self.gru_layers = nn.ModuleList([self.GRUCell(self.hidden_size, self.hidden_size) for _ in range(self.num_layers)])
        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )
        
        self.model_path = model_path
        self.load_saved_model()

    def forward(self, x):
        batch_size, seq_len, feature_size = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        # Pass through initial linear layer
        x = self.initial_layer(x)
        
        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in self.gru_layers:
                h = layer(x_t, h)
        
        # Pass through final output layer
        y = self.final_layer(h)
        return y
 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResidualBlockGroup(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride):
        super(ResidualBlockGroup, self).__init__()
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        self.group = nn.Sequential(*layers)

    def forward(self, x):
        return self.group(x)


class ResNetModel(BaseModel):
    def __init__(self, input_shape, out_classes, config, model_path):
        super(ResNetModel, self).__init__()
        self.method = 'ResNet'
        self.config = config
        
        layers_config = config['model']['layers']
        self.layers = self.make_layers(config['model']['layers'])

        self.model = nn.Sequential(*self.layers)
        self.model_path = model_path
        self.load_saved_model()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)
        k = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class MaxLayer(nn.Module):
    def __init__(self):
        super(MaxLayer, self).__init__()

    def forward(self, x):
        max_value, _ = torch.max(x, dim=1)
        return max_value


class TransformerModel(BaseModel):
    def __init__(self, input_size, output_size, config, model_path=None):
        super().__init__()
        self.method = 'Transformer'
        self.config = config
        self.layers = self.make_layers(self.config['model']['layers'])
        self.model = nn.Sequential(*self.layers)
        self.model_path = model_path
        self.load_saved_model()

    def forward(self, x, **kwargs):
        mask = kwargs['mask']
        if 'TransformerEncoderLayer' in [layer['type'] for layer in self.config['model']['layers']]:
            for module in self.model:
                if isinstance(module, nn.TransformerEncoderLayer):
                    x = module(x, src_key_padding_mask=mask)
                else:
                    x = module(x)
        else:
            x = self.model(x)
        return x
            

class WeibullCoxModel(BaseModel):
    def __init__(self, model_path=None):
        super().__init__()
        self.method = 'WeibullCox'
        if model_path:
            with open(model_path, 'r') as fp:
                wc_model = json.load(fp)
            self.lambda_raw = nn.Parameter(torch.tensor([wc_model['lambda']], requires_grad=True))
            self.k_raw = nn.Parameter(torch.tensor([wc_model['k']], requires_grad=True))
            self.beta = nn.Parameter(torch.tensor(wc_model['beta'], requires_grad=True))
        else:
            self.lambda_raw = nn.Parameter(torch.tensor([0.0], requires_grad=True))
            self.k_raw = nn.Parameter(torch.tensor([0.0], requires_grad=True))
            self.beta = nn.Parameter(torch.randn(40, requires_grad=True))
            
    def log_likelihood(self, x, tau, S, pos_weight):
        lambda_ = F.softplus(self.lambda_raw)
        k = F.softplus(self.k_raw)
        beta = self.beta
        tau_lambda_ratio = tau / lambda_
        exp_beta_x = torch.exp(torch.matmul(x, beta))
        likelihood = (1 - S) * pos_weight * (torch.log(k) - torch.log(lambda_) + (k - 1) * torch.log(tau_lambda_ratio) + torch.matmul(x, beta)) - (tau_lambda_ratio ** k) * exp_beta_x
        return likelihood.sum()

    def forward(self, x, **param):
        tau = 6
        x = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
        lambda_ = F.softplus(self.lambda_raw)
        k = F.softplus(self.k_raw)
        beta = self.beta
        tau_lambda_ratio = tau / lambda_
        exp_beta_x = torch.exp(torch.matmul(x, beta))
        probs = 1 - torch.exp(-(tau_lambda_ratio ** k) * exp_beta_x)

        return probs.squeeze()
    
    def train_model(self, dataset, use_val=False, epochs=50, batch_size=256, pos_weight=54.5, lr=0.001, loss_criterion='BCE', logging=False, num_workers=24):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam([self.lambda_raw, self.k_raw, self.beta], lr=lr)
        
        # start a new wandb run to track this script
        if logging:
            run = wandb.init(
                # set the wandb project where this run will be logged
                project="sepsis-binary-classification",
    
                # track hyperparameters and run metadata
                config={
                    "architecture"       : "Weibull Cox",
                    "dataset"            : "Competition2019",
                    "preprocessing"      : "standardized",
                    "batch_size"         : batch_size,
                    "learning_rate"      : lr,
                    "epochs"             : epochs,
                    "pos_weight"         : pos_weight
                }
            )
            rid = run.name
        else:
            rid = self.method + '_trial'
        
        for epoch in range(epochs):
            total_loss = 0
            # patient_id, latest_hour, clinical_data, label, utility_weights, tau, S
            for _, _, x_batch, _, _, tau_batch, S_batch in tqdm(train_loader, desc="Epoch {}".format(epoch), ascii=False, ncols=75):
                optimizer.zero_grad()
                loss = -self.log_likelihood(x_batch, tau_batch, S_batch, pos_weight)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}, Loss: {epoch_loss}')
            if logging:
                wandb.log({
                    "Train loss" : epoch_loss
                })

        now = datetime.now()
        timestr = now.strftime("%m_%d_%Y_%H_%M_%S")
        model_path = '../models/{}_{}_{:.5f}_{}.pth'.format(rid, epoch, epoch_loss, timestr)
        with open(model_path, "w") as fp:
            json.dump({'lambda':self.lambda_raw.item(),'k':self.k_raw.item(), 'beta':self.beta.tolist()}, fp)
            print('Model saved at:{}'.format(model_path))

        wandb.finish()
        return rid