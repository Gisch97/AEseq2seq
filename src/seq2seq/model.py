import pandas as pd
import math
from dataclasses import dataclass
from torch import nn
from torch.nn.functional import mse_loss
import torch as tr
from tqdm import tqdm
 
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
        
    return model
    
    
class Seq2Seq(nn.Module):
    def __init__(self,
        train_len=0,
        embedding_dim=4,
        device="cpu",
        negative_weight=0.1,
        lr=1e-4,
        loss_l1=0,
        loss_beta=0,
        scheduler="none",
        verbose=True,
        interaction_prior=False,
        output_th=0.5,
        **kwargs):
        """Base instantiation of model"""
        super().__init__()

        self.device = device
        self.class_weight = tr.tensor([negative_weight, 1.0]).float().to(device)
        self.verbose = verbose
        self.config = kwargs
        self.output_th = output_th

        mid_ch = 1
        self.interaction_prior = interaction_prior
        if interaction_prior != "none":
            mid_ch = 2

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
        filters=32,
        kernel=3,
        num_layers=2,
        dilation_resnet1d=3,
        resnet_bottleneck_factor=0.5,
        latent_dim=16,
        rank=64,
        **kwargs
    ): 
        pad = (kernel - 1) // 2
        # Encoder
        self.encode = [nn.Conv1d(embedding_dim, filters, kernel, padding="same")]
        for k in range(num_layers):
            self.encode.append(
                ResidualLayer1D(
                    dilation_resnet1d,
                    resnet_bottleneck_factor,
                    filters,
                    kernel,
                )
            )
        self.encode.append(
            nn.Conv1d(
                in_channels=filters,
                out_channels=rank,
                kernel_size=kernel,
                padding=pad,
                stride=1,
                )
            )
        self.encode = nn.Sequential(*self.encode)
        
        self.to_latent = nn.Sequential(nn.Flatten(1),
                                        nn.Linear(128 * 64, latent_dim),
                                        nn.ReLU())
        
        
        # Decoder 
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, 128 * 64),
                                          nn.ReLU())

        self.decode = [nn.ConvTranspose1d(
            in_channels=rank,
            out_channels=filters,
            kernel_size=kernel,
            padding=pad,
            stride=1
            )]
        for k in range(num_layers):
            self.decode.append(
                ResidualLayer1D(
                    dilation_resnet1d,
                    resnet_bottleneck_factor,
                    filters,
                    kernel,
                )
            )
        self.decode.append(nn.Conv1d(filters, embedding_dim, kernel, padding="same"))
        self.decode = nn.Sequential(*self.decode)


    def forward(self, batch):
        x = batch["embedding"].to(self.device)
        batch_size = x.shape[0]
        L = x.shape[2]
        
        z = self.encode(x) 
        z = self.to_latent(z)
        x_rec = self.from_latent(z)

        x_rec = x_rec.view(x_rec.shape[0], -1, L)
        x_rec = self.decode(x_rec)
        return x_rec, z

    def loss_func(self, x_rec, x):
        """yhat and y are [N, L]"""
        x = x.view(x.shape[0], -1)
        x_rec = x_rec.view(x_rec.shape[0], -1)

        loss = mse_loss(x_rec, x)
    
        return loss

    def fit(self, loader):
        self.train()
        metrics = {"loss": 0}

        if self.verbose: loader = tqdm(loader)

        for batch in loader: 
            x = batch["embedding"].to(self.device)
            # batch.pop("embedding")
            self.optimizer.zero_grad()  # Cleaning cache optimizer
            x_rec, z = self(batch)
            
            loss = self.loss_func(x_rec, x) 

            metrics["loss"] += loss.item()

            loss.backward()
            self.optimizer.step()

            if self.scheduler_name == "cycle":
                    self.scheduler.step()

        for k in metrics:
            metrics[k] /= len(loader)

        return metrics

    def test(self, loader):
        self.eval()
        metrics = {"loss": 0}

        if self.verbose:
            loader = tqdm(loader)

        with tr.no_grad():
            for batch in loader:  
                x = batch["embedding"].to(self.device)
                # batch.pop("embedding")
                lengths = batch["length"]
                
                x_rec, z = self(batch)
                loss = self.loss_func(x_rec, x)
                metrics["loss"] += loss.item()

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
    
