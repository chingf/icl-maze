import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch.multiprocessing as mp
from transformers import GPT2Config, GPT2Model
from src.models.transformer import Transformer
from IPython import embed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ZeroEmbedding(nn.Module):
    def forward(self, position_ids):
        return 0

class SingleLayerTransformer(Transformer):
    """Transformer class."""

    def __init__(
        self,
        n_embd: int,
        n_embed_layer: int,
        n_out_layer: int,
        n_head: int,
        state_dim: int,
        action_dim: int,
        dropout: float,
        train_on_last_pred_only: bool,
        test: bool,
        name: str,
        optimizer_config: dict,
        ):

        super(Transformer, self).__init__()

        self.n_embd = n_embd
        self.n_embed_layer = n_embed_layer
        self.n_transformer_layers = 1
        self.n_out_layer = n_out_layer
        self.n_head = n_head
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout = dropout
        self.train_on_last_pred_only = train_on_last_pred_only
        self.test = test
        self.optimizer_config = optimizer_config = optimizer_config

        config = GPT2Config(
            n_positions=1000,  # Arbitrary, as position embeddings are not used
            n_embd=self.n_embd,
            n_layer=self.n_transformer_layers,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)
        self.remove_unused_params(self.transformer)
        self.transformer.wpe = ZeroEmbedding()
        self.embed_transition = self.make_embed_mlp_layers()
        self.pred_actions = self.make_out_mlp_layers()
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

    def make_embed_mlp_layers(self):
        layers = []
        in_size = 2 * self.state_dim + self.action_dim + 1
        for layer in range(self.n_embed_layer):
            if layer == self.n_embed_layer - 1:
                out_size = self.n_embd
            else:
                out_size = min(in_size*2, self.n_embd)
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            in_size = out_size
        return nn.Sequential(*layers)

    def make_out_mlp_layers(self):
        layers = []
        in_size = self.n_embd
        for layer in range(self.n_out_layer):
            if layer == self.n_out_layer - 1:
                out_size = self.action_dim
            else:
                out_size = max(in_size//2, 32)
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            in_size = out_size
        return nn.Sequential(*layers)

    def forward(self, x):
        query_states = x['query_states'][:, None, :]
        zeros = x['zeros'][:, None, :]

        state_seq = torch.cat([query_states, x['context_states']], dim=1)
        action_seq = torch.cat(
            [zeros[:, :, :self.action_dim], x['context_actions']], dim=1)
        next_state_seq = torch.cat(
            [zeros[:, :, :self.state_dim], x['context_next_states']], dim=1)
        reward_seq = torch.cat([zeros[:, :, :1], x['context_rewards']], dim=1)

        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        stacked_inputs = self.embed_transition(seq)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        transformer_outputs = transformer_outputs['last_hidden_state']  # Last (and only) layer
        batch_size, seq_len, token_dim = transformer_outputs.shape
        transformer_outputs = transformer_outputs.reshape(batch_size * seq_len, token_dim)
        preds = self.pred_actions(transformer_outputs)
        preds = preds.reshape(batch_size, seq_len, -1)

        if self.test:
            return preds[:, -1, :]
        if self.train_on_last_pred_only:
            indices = torch.tensor([preds.shape[1] - 1]).to(preds.device)
        else:
            indices = torch.cat([
                torch.tensor([0]), torch.arange(10, preds.shape[1], 10)]
            ).to(preds.device)
        preds_subset = torch.index_select(preds, 1, indices)
        return preds_subset

    def batch_forward(self, batch, batch_idx):
        pred_actions = self(batch)
        true_actions = batch['optimal_actions']
        
        # Reshape predictions and targets for loss calculation
        true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)
        true_actions = true_actions.reshape(-1, self.action_dim)
        pred_actions = pred_actions.reshape(-1, self.action_dim)
        # Calculate accuracy by comparing predicted and true actions
        horizon = pred_actions.shape[0]
        pred_actions_idx = torch.argmax(pred_actions, dim=1) 
        true_actions_idx = torch.argmax(true_actions, dim=1)
        accuracy = (pred_actions_idx == true_actions_idx).float().mean()
        loss = self.loss_fn(pred_actions, true_actions) / horizon
        return loss, accuracy

  
    def training_step(self, batch, batch_idx):
        batch_size = batch['query_states'].shape[0]
        loss, accuracy = self.batch_forward(batch, batch_idx)
        self.log(
            'train_loss', loss/batch_size,
            on_epoch=True, on_step=False, prog_bar=True)
        self.log(
            'train_accuracy', accuracy,
            on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['query_states'].shape[0]
        loss, accuracy = self.batch_forward(batch, batch_idx)
        self.log(
            'val_loss', loss/batch_size,
            on_epoch=True, on_step=False, prog_bar=True)
        self.log(
            'val_accuracy', accuracy,
            on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config['lr'],
            weight_decay=self.optimizer_config['weight_decay']
        )
        lr_scheduler = {  # linearly decrease LR from 1e-3 to 1e-4 over 75 epochs
            'scheduler': torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,  # Start at 1e-3 (10x higher than final 1e-4) 
                end_factor=0.1,     # End at 1e-4
                total_iters=50,      # Linear decrease over 50 epochs
            ),
            'monitor': 'val_loss',
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    
    def remove_unused_params(self, model):
        try:
            delattr(model, 'wpe')
        except:
            print('Attempted to delete WPE in transformer, but could not find.')
        try:
            delattr(model, 'wte')
        except:
            print('Attempted to delete WTE in transformer, but could not find.')
