import pandas as pd
import math
from dataclasses import dataclass
from torch import nn
from torch.nn.functional import mse_loss
import torch as tr
from tqdm import tqdm
 
from .utils import mat2bp, postprocessing
from ._version import __version__


@dataclass
class ModelConfig:
    """Configuration class for Seq2Seq model hyperparameters"""
    
    # Configuración del sistema
    device: str = "cpu"
    verbose: bool = True
    
    # Parámetros de entrenamiento
    lr: float = 1e-4
    negative_weight: float = 0.1
    output_th: float = 0.5
    scheduler: str = "none"
    interaction_prior: bool = False 
    
    # Parámetros generales del modelo
    embedding_dim: int = 4
    hidden_dim: int = 32
    latent_dim: int = 4
    
    # Parámetros de la arquitectura
    num_layers: int = 2
    kernel: int = 3
    filters: int = 32
    rank: int = 64
    mid_ch: int = 1
    dilation_resnet1d: int = 3
    resnet_bottleneck_factor: float = 0.5
    
def seq2seq(weights=None, **kwargs): 
    """ 
    seq2seq: a deep learning-based autoencoder for RNA sequence to sequence prediction.
    weights (str): Path to weights file
    **kwargs: Model hyperparameters
    """
    
    model = Seq2Seq(c, **kwargs)
    if weights is not None:
        print(f"Load weights from {weights}")
        model.load_state_dict(tr.load(weights, map_location=tr.device(model.device)))
    else:
        print("No weights provided, using random initialization")
        
    return model
    
    
class Seq2Seq(nn.Module):
    def __init__(self, c: ModelConfig, **kwargs):
        """Base instantiation of model"""
        super().__init__()

        self.device = c.device
        self.class_weight = tr.tensor([c.negative_weight, 1.0]).float().to(c.device)
        self.verbose = c.verbose
        self.config = kwargs
        self.output_th = c.output_th

        mid_ch = 1
        self.interaction_prior = c.interaction_prior
        if c.interaction_prior != "none":
            mid_ch = 2

        # Define architecture
        self.build_graph(c, **kwargs) # encoder / decoder
        self.optimizer = tr.optim.Adam(self.parameters(), lr=c.lr)

        # lr scheduler
        self.scheduler_name = c.scheduler
        if c.scheduler == "plateau":
            self.scheduler = tr.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, verbose=True
            )
        elif c.scheduler == "cycle":
            self.scheduler = tr.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=c.lr, steps_per_epoch=c.train_len, epochs=self.config["max_epochs"]
            )
        else:
            self.scheduler = None

        self.to(device)
    
    def build_graph(self, c: ModelConfig, **kwargs):
        pad = (c.kernel - 1) // 2
        
        self.resnet1d = [nn.Conv1d(c.embedding_dim, c.filters, c.kernel, padding="same")]
        for k in range(c.num_layers):
            self.resnet1d.append(
                ResidualLayer1D(
                    c.dilation_resnet1d,
                    c.resnet_bottleneck_factor,
                    c.filters,
                    c.kernel,
                )
            )
            
        self.resnet1d = nn.Sequential(*self.resnet1d)
        self.convrank1 = nn.Conv1d(
            in_channels=c.filters,
            out_channels=c.rank,
            kernel_size=c.kernel,
            padding=pad,
            stride=1,
        )
        self.encoder = nn.Sequential(*self.resnet1d, self.convrank1)

    def forward(self, batch):
        x = batch["embedding"].to(self.device)
        batch_size = x.shape[0]
        L = x.shape[2]
        
        y = self.resnet1d(x)
        y = y.view(-1, L) 
        y = tr.relu(y).squeeze(1)
         

        return y

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
            
            x = batch["contact"].to(self.device)
            batch.pop("contact")
            self.optimizer.zero_grad()  # Cleaning cache optimizer
            x_rec = self(batch)
            
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
                x = batch["contact"].to(self.device)
                batch.pop("contact")
                lengths = batch["length"]
                
                x_rec = self(batch)
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
                
                lengths = batch["length"]
                seqid = batch["id"]
                sequences = batch["sequence"]
                x_rec = self(batch)
                
                # x_rec_post = postprocessing(x_rec.cpu(), batch["canonical_mask"])

                for k in range(x_rec.shape[0]):
                    if logits:
                        logits_list.append(
                            (seqid[k],
                             x_rec[k, : lengths[k]].squeeze().cpu()
                            ))
                    predictions.append(
                        (seqid[k],
                        sequences[k],
                                x_rec[k, : lengths[k]].squeeze()
                        )
                    )
        predictions = pd.DataFrame(predictions, columns=["id", "sequence","reconstructed"])

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
    
class Encoder(nn.Module):
    """RNA sequence encoder using 1D convolutions and residual connections"""

    def __init__(self, c: ModelConfig):
        super().__init__()
         
        layers = [nn.Conv1d(c.embedding_dim, c.hidden_dim, 
                            c.kernel_size, padding="same")]
         
        for _ in range(c.num_layers):
            layers.append(ResidualLayer1D(
                    c.dilation_resnet1d,
                    c.resnet_bottleneck_factor,
                    c.filters,
                    c.kernel))
        
        self.encoder_layers = nn.Sequential(*layers)
         
        self.projection = nn.Conv1d(
            c.hidden_dim,
            c.latent_dim,
            c.kernel_size,
            padding="same"
        )

    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """
        Forward pass through encoder
        Args:
            x: Input tensor of shape [batch_size, embedding_dim, seq_length]
        Returns:
            Encoded representation of shape [batch_size, latent_dim, seq_length]
        """
        x = self.encoder_layers(x)
        return self.projection(x)
    
class Decoder(nn.Module):
    """RNA sequence decoder that reconstruye la secuencia a partir de la representación codificada"""
    
    def __init__(self, c: ModelConfig):
        super().__init__()
        pad = (c.kernel_size - 1) // 2
        # Feed-forward network para procesar la entrada
        self.input_ff = nn.Sequential(
            nn.Linear(c.latent_dim, c.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim * 4, c.hidden_dim)
        )
        
        # Capas de deconvolución
        decoder_layers = []
        
        # Capa inicial de deconvolución
        decoder_layers.append(
            nn.ConvTranspose1d(
                c.hidden_dim,
                c.hidden_dim,
                c.kernel_size,
                padding=pad
            )
        )
        
        # Bloques residuales
        for _ in range(c.num_layers):
            decoder_layers.append(ResidualBlock1D(c))
        
        self.decoder_layers = nn.Sequential(*decoder_layers)
        
        # Capa de proyección final
        self.final_projection = nn.Sequential(
            nn.ConvTranspose1d(
                c.hidden_dim,
                c.embedding_dim,
                c.kernel_size,
                padding=pad
            ),
            nn.ReLU()
        )
    
    def forward(self, x: tr.Tensor) -> tr.Tensor:
        """
        Forward pass a través del decoder
        Args:
            x: Tensor codificado de forma [batch_size, latent_dim, seq_length]
        Returns:
            Secuencia reconstruida de forma [batch_size, embedding_dim, seq_length]
        """
        # Aplicar feed-forward
        batch_size, latent_dim, seq_length = x.shape
        x = x.transpose(1, 2)  # [batch_size, seq_length, hidden_dim]
        x = self.input_ff(x)
        x = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_length]
        
        # Aplicar capas de deconvolución
        x = self.decoder_layers(x)
        
        # Proyección final
        output = self.final_projection(x)
        
        return output
