import pandas as pd
import math 
from torch import nn 
from torch.nn.functional import mse_loss, cross_entropy
import torch as tr
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from ..conv_layers import N_Conv, Up_Block, Max_Down, OutConv 
from ...metrics import compute_metrics 
from ..._version import __version__


    
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
        kernel=3,
        num_conv=2,
        up_mode = 'traspose',
        addition='cat',
        skip=False,
        **kwargs
    ):     
        
        self.features = [4, 8, 8, 8]
        self.r_features = self.features[::-1]
        self.encoder_blocks = len(self.features) - 1
        self.L_min = 128 // ((2 ** self.encoder_blocks))
        volume = [(128 / 2 ** i) * f for i, f in enumerate(self.features)]
        
        self.architecture = {
            "arc_embedding_dim": embedding_dim,
            "arc_encoder_blocks": self.encoder_blocks,
            "arc_initial_volume": embedding_dim * 128,
            "arc_latent_volume": volume[-1],
            "arc_features": self.features,
            "arc_num_conv": num_conv,
            "arc_up_mode": up_mode,
            "arc_addition": addition,
            "arc_skip": skip,
        }
        linear = True
        self.inc = (N_Conv(embedding_dim, self.features[0], num_conv))
        
        self.down = nn.ModuleList(
            [
                Max_Down(self.features[i], self.features[i + 1], num_conv)
                for i in range(self.encoder_blocks)
            ]
        )
        self.up = nn.ModuleList(
            [
                Up_Block( in_channels = self.r_features[i], 
                          out_channels = self.r_features[i+1],
                          num_conv = num_conv,
                          up_mode = up_mode,
                          addition = addition,
                          skip = skip
                         )
                for i in range(len(self.r_features) - 1)
            ]
        ) 
        self.outc = OutConv(self.features[0], embedding_dim)
        
    def forward(self, batch):
        x = batch["embedding"].to(self.device)

        x = self.inc(x)
        encoder_outputs = [x] 
        for i, down in enumerate(self.down):
            x = down(x)
            encoder_outputs.append(x) 

        x_latent = x
         
        skips = encoder_outputs[:-1][::-1]
        for up, skip in zip(self.up, skips):
            x = up(x, skip)

        x_rec = self.outc(x)

        return x_rec, x_latent

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
            self.optimizer.zero_grad()  # Cleaning cache optimizer
            x_rec, _ = self(batch)
            loss = self.loss_func(x_rec, x) 
            metrics["loss"] += loss.item()
            
            ce_loss = self.ce_loss_func(x_rec, x)
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
