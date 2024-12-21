import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch.multiprocessing as mp
from IPython import embed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(pl.LightningModule):
    """LSTM-based model."""

    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        state_dim: int,
        action_dim: int,
        dropout: float,
        train_on_last_pred_only: bool,
        test: bool,
        name: str,
        optimizer_config: dict,
        ):

        super(RNN, self).__init__()

        self.n_embd = n_embd
        self.n_layer = n_layer
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout = dropout
        self.train_on_last_pred_only = train_on_last_pred_only
        self.test = test
        self.optimizer_config = optimizer_config = optimizer_config

        print(f"LSTM Dropout: {self.dropout}")
        self.rnn = torch.nn.LSTM(
            input_size=self.n_embd,
            hidden_size=self.n_embd,  # Consider expansion
            num_layers=self.n_layer,
            batch_first=True,
            dropout=self.dropout,
            )
        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim + 1, self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

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
        rnn_outputs, _ = self.rnn(stacked_inputs)  # (batch, seq_len, hidden_size)
        preds = self.pred_actions(rnn_outputs)

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
                end_factor=0.01,     # End at 1e-4
                total_iters=100,      # Linear decrease over 50 epochs
            ),
            'monitor': 'val_loss',
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}