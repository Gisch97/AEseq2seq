import pandas as pd
import math
from dataclasses import dataclass
from torch import nn
from torchinfo import summary
from torch.nn.functional import mse_loss, cross_entropy
import torch as tr
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from .metrics import compute_metrics
from .utils import mat2bp, postprocessing
from ._version import __version__


    
def seq2seq(weights=None, **kwargs): 
    """ 
    seq2seq: a deep learning-based autoencoder for RNA sequence to sequence prediction.
    weights (str): Path to weights file
    **kwargs: Model hyperparameters
    """
    
    model = Seq2Seq(**kwargs)
    if weights is not None:
        print(f"Load weights from {weights}")
        model.load_state_dict(tr.load(weights, map_location=tr.device(model.device)))
    else:
        print("No weights provided, using random initialization")
    model.log_model()
    mlflow.set_tag("model", 'Unet')
    return model
    
    
class Seq2Seq(nn.Module):
    def __init__(self,
        train_len=0,
        embedding_dim=4,
        device="cpu", 
        lr=1e-3,
        scheduler="none",
        output_th=0.5,
        verbose=True,
        **kwargs):
        """Base instantiation of model"""
        super().__init__()


        self.device = device
        self.verbose = verbose
        self.config = kwargs
        self.output_th = output_th
        
        self.hyperparameters = {
            "hyp_device": device, 
            "hyp_lr": lr,
            "hyp_scheduler": scheduler,
            "hyp_verbose": verbose, 
            "hyp_output_th": output_th
            }        
        # Define architecture
        self.build_graph(embedding_dim, **kwargs) 
        self.optimizer = tr.optim.Adam(self.parameters(), lr=lr)

        # lr scheduler
        self.scheduler_name = scheduler
        if scheduler == "plateau":
            self.scheduler = tr.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, verbose=True
            )
        elif scheduler == "cycle":
            self.scheduler = tr.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=lr, steps_per_epoch=train_len, epochs=self.config["max_epochs"]
            )
        else:
            self.scheduler = None

        self.to(device)
    
    def build_graph(
        self,
        embedding_dim,
        filters=4,
        kernel=3,
        num_layers=2,
        dilation_resnet1d=3,
        resnet_bottleneck_factor=0.5,
        rank=8,
        stride_1=2, 
        stride_2=2,
        num_conv1=3,
        num_conv2=3,
        **kwargs
    ):     
            
         # Encoder: 
        self.encode1_in = embedding_dim
        self.encode1_out = 4
        self.encode2_in = self.encode1_out
        self.encode2_out =  8
   
        # Decoder
        self.decode1_in = self.encode2_out  
        self.decode1_out = self.encode2_in  
        
        self.decode2_in = self.encode1_out
        self.decode2_out = self.encode1_in
        self.L_min = 128 // ((2 ** num_conv2) * (2**num_conv1))
        self.latent_dim=64
          
        
        self.architecture = {
            "arc_embedding_dim": embedding_dim,
            "arc_filters": self.encode1_out,
            "arc_rank": self.encode2_out,
            "arc_latent_dim": self.latent_dim,
            "arc_initial_volume": embedding_dim * 128,
            "arc_latent_volume": self.L_min * self.encode2_out,
            "arc_kernel": kernel,
            "arc_num_layers": num_layers,
            "arc_dilation_resnet1d": dilation_resnet1d,
            "arc_resnet_bottleneck_factor": resnet_bottleneck_factor,
            "arc_stride_1": stride_1,
            "arc_stride_2": stride_2,
            "arc_num_conv1": num_conv1,
            "arc_num_conv2": num_conv2
        }
        pad = (kernel - 1) // 2

        self.encode1 = conv_sequence_avg_pooled(
            input_channels=self.encode1_in, 
            output_channels=self.encode1_out, 
            num_conv=num_conv1,  
            padding=pad
        )

        self.encode2 = nn.Sequential(
            *[ResidualLayer1D(
                dilation_resnet1d,
                resnet_bottleneck_factor,
                self.encode1_out,
                kernel
            ) for _ in range(num_layers)],
            conv_sequence_avg_pooled(  
                input_channels=self.encode2_in, 
                output_channels=self.encode2_out, 
                num_conv=num_conv2, 
                padding=pad
            )
        )

        self.decode1 = transpose_conv_sequence(
            input_channels=self.decode1_in,
            output_channels=self.decode1_out,
            num_conv=num_conv2, 
            kernel_size=kernel, 
            padding=pad, 
            stride=stride_2)
        self.decode2 = nn.Sequential(
            *[ResidualLayer1D(
                dilation_resnet1d,
                resnet_bottleneck_factor,
                self.decode2_in,
                kernel
            ) for _ in range(num_layers)],
            transpose_conv_sequence(
                input_channels=self.decode2_in,  
                output_channels=self.decode2_out,
                num_conv=num_conv1,  
                kernel_size=kernel, 
                padding=pad, 
                stride=stride_1)
        )
        
        
        self.to_latent = nn.Sequential(nn.Flatten(1),
                                        nn.Linear(self.encode2_out * self.L_min, self.latent_dim),
                                        nn.ReLU())
        
        
        # Decoder 
        self.from_latent = nn.Sequential(nn.Linear(self.latent_dim, self.encode2_out * self.L_min),
                                          nn.ReLU(),
                                          nn.Unflatten(1, (self.encode2_out, self.L_min)))
    def forward(self, batch): 
        
        x = batch["embedding"].to(self.device) 
        x1 = self.encode1(x)  
        x2 = self.encode2(x1)  
        z = self.to_latent(x2)  
        x3 = self.from_latent(z)  
        x4 = self.decode1(x3)  
        x_rec = self.decode2(x4)  
        return x_rec, z
        
    def loss_func(self, x_rec, x):
        """yhat and y are [N, L]"""
        x = x.view(x.shape[0], -1)
        x_rec = x_rec.view(x_rec.shape[0], -1)
        recon_loss = mse_loss(x_rec, x) 
        return recon_loss   

    
    def ce_loss_func(self, x_rec, x):
        """yhat and y are [N, L]"""
        x = x.view(x.shape[0], -1)
        x_rec = x_rec.view(x_rec.shape[0], -1)
        loss = cross_entropy(x_rec, x)
        return loss


    def fit(self, loader):
        self.train()

        metrics = {
            "loss": 0,
            "ce_loss": 0,
            "F1": 0,
            "Accuracy": 0,
            "Accuracy_seq": 0
            }
        if self.verbose: loader = tqdm(loader)

        for batch in loader: 
            x = batch["embedding"].to(self.device)
            # batch.pop("embedding")
            self.optimizer.zero_grad()  # Cleaning cache optimizer
            x_rec, z = self(batch)
            loss = self.loss_func(x_rec, x) 
            ce_loss = self.ce_loss_func(x_rec, x)
            metrics["loss"] += loss.item()
            metrics["ce_loss"] += ce_loss.item()
            
            
            batch_metrics = compute_metrics(x_rec, x, output_th=self.output_th)
            for k, v in batch_metrics.items():
                metrics[k] += v
            
            
            loss.backward()
            self.optimizer.step()

            if self.scheduler_name == "cycle":
                    self.scheduler.step()

        for k in metrics:
            metrics[k] /= len(loader)

        return metrics

    def test(self, loader):
        self.eval()
        
        metrics = {
            "loss": 0,
            "ce_loss": 0,
            "F1": 0,
            "Accuracy": 0,
            "Accuracy_seq": 0
            }

        if self.verbose:
            loader = tqdm(loader)

        with tr.no_grad():
            for batch in loader:  
                x = batch["embedding"].to(self.device)
                # batch.pop("embedding")
                lengths = batch["length"]
                
                x_rec, z = self(batch)
                loss = self.loss_func(x_rec, x)
                ce_loss = self.ce_loss_func(x_rec, x)
                metrics["loss"] += loss.item()
                metrics["ce_loss"] += ce_loss.item()
                
                
                batch_metrics = compute_metrics(x_rec, x, output_th=self.output_th)
                for k, v in batch_metrics.items():
                    metrics[k] += v

        for k in metrics: metrics[k] /= len(loader)

        return metrics

    def pred(self, loader, logits=False):
        self.eval()

        if self.verbose:
            loader = tqdm(loader)

        predictions, logits_list = [], [] 
        with tr.no_grad():
            for batch in loader: 
                
                seqid = batch["id"]
                embedding = batch["embedding"]
                sequences = batch["sequence"]
                lengths = batch["length"]
                x_rec, z = self(batch)
                
                for k in range(x_rec.shape[0]):
                    seq_len = lengths[k]
                
                    predictions.append((
                        seqid[k],
                        sequences[k],
                        seq_len,
                        embedding[k, :, :seq_len].cpu().numpy(),
                        x_rec[k, :, :seq_len].cpu().numpy(),
                        z[k].cpu().numpy()
                    ))
                    
        predictions = pd.DataFrame(predictions, columns=["id", "sequence", "length", "embedding", "reconstructed", "latent"])

        return predictions, logits_list

    def log_model(self):
        """Logs the model architecture and hyperparameters to MLflow.""" 
        mlflow.log_params(self.hyperparameters)
        mlflow.log_params(self.architecture)

        # with open("model_summary.txt", "w") as f:
        #     f.write(str(summary(self)))
        # mlflow.log_artifact("model_summary.txt")

class ResidualLayer1D(nn.Module):
    def __init__(
        self,
        dilation,
        resnet_bottleneck_factor,
        filters,
        kernel_size,
    ):
        super().__init__()

        num_bottleneck_units = math.floor(resnet_bottleneck_factor * filters)

        self.layer = nn.Sequential(
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Conv1d(
                filters,
                num_bottleneck_units,
                kernel_size,
                dilation=dilation,
                padding="same",
            ),
            nn.BatchNorm1d(num_bottleneck_units),
            nn.ReLU(),
            nn.Conv1d(num_bottleneck_units, filters, kernel_size=1, padding="same"),
        )

    def forward(self, x):
        return x + self.layer(x)
    

def conv_sequence_avg_pooled(input_channels, output_channels, num_conv=1,  padding=1, pool_stride=2, pool_kernel=2): 
    
    layers = [] 
    layers.append(nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=padding, stride=1))
    layers.append(nn.BatchNorm1d(output_channels)) 
    layers.append(nn.ReLU(inplace=True)) 
    layers.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride, padding=0)) 
    
    for _ in range(num_conv-1):
        layers.append(nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=padding, stride=1))    
        layers.append(nn.BatchNorm1d(output_channels))
        layers.append(nn.ReLU(inplace=True))     
        layers.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride, padding=0)) 
    return nn.Sequential(*layers)

def transpose_conv_sequence(input_channels, output_channels, num_conv, kernel_size, padding, stride):
    layers = []
    for _ in range(num_conv - 1):
        layers.append(nn.ConvTranspose1d(input_channels, input_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1)) 
        layers.append(nn.BatchNorm1d(input_channels)) 
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=stride - 1)) 
    layers.append(nn.BatchNorm1d(output_channels)) 
    layers.append(nn.ReLU(inplace=True))
    
    return nn.Sequential(*layers)