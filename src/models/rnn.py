import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch.multiprocessing as mp
from IPython import embed
from transformers import set_seed
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
        test: bool,
        name: str,
        optimizer_config: dict,
        initialization_seed: int,
        train_query_every: int=10,
        ):

        super(RNN, self).__init__()

        self.n_embd = n_embd
        self.n_layer = n_layer
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout = dropout
        self.test = test
        self.optimizer_config = optimizer_config = optimizer_config
        self.initialization_seed = initialization_seed
        self.train_query_every = train_query_every
        set_seed(self.initialization_seed)

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
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        #torch.nn.Sequential(
        #    nn.Linear(self.n_embd, self.n_embd//2),
        #    nn.ReLU(),
        #    nn.Linear(self.n_embd//2, self.action_dim)
        #)

    def forward(self, x):
        query_states = x['query_states'][:, None, :]
        zeros = x['zeros'][:, None, :]

        # Prepare context sequence
        state_seq = x['context_states']
        action_seq = x['context_actions']
        next_state_seq = x['context_next_states']
        reward_seq = x['context_rewards']
        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        stacked_inputs = self.embed_transition(seq)
        _, seq_len, _ = stacked_inputs.shape

        # Prepare query
        query = torch.cat([query_states, zeros[:,:,:self.action_dim+self.state_dim+1]], dim=2)
        query = self.embed_transition(query)
        query_every = self.train_query_every

        # Run through RNN
        rnn_hidden = None
        rnn_query_outputs = []
        for i in range(seq_len):
            rnn_outputs, rnn_hidden = self.rnn(stacked_inputs[:, i:i+1, :], rnn_hidden)
            if i % query_every == 0 or i == seq_len - 1:
                _rnn_query_output, _ = self.rnn(query, rnn_hidden)
                rnn_query_outputs.append(_rnn_query_output)
        rnn_query_outputs = torch.cat(rnn_query_outputs, dim=1)
        preds = self.pred_actions(rnn_query_outputs)

        if self.test:
            return preds[:, -1, :]
        return preds

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
        return {'optimizer': optimizer} #, 'lr_scheduler': lr_scheduler}
