import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.multiprocessing as mp
import transformers
transformers.set_seed(0)
from transformers import GPT2Config
from src.models.simple_gpt2 import SimpleGPT2Model
from src.models.fixedmemory_gpt2 import FixedMemoryGPT2Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Transformer(pl.LightningModule):
    """Transformer class."""

    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        n_head: int,
        state_dim: int,
        action_dim: int,
        dropout: float,
        train_on_last_pred_only: bool,
        test: bool,
        name: str,
        optimizer_config: dict,
        linear_attention: bool = False,
        ):

        super(Transformer, self).__init__()

        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout = dropout
        self.train_on_last_pred_only = train_on_last_pred_only
        self.test = test
        self.optimizer_config = optimizer_config = optimizer_config
        self.linear_attention = linear_attention

        config = GPT2Config(
            n_positions=2000,  # Position embeddings are not used, but make sure it's > seq length for inference mode
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = SimpleGPT2Model(config, linear_attention)
        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim + 1, self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.save_activations = False
        self.pass_query_locations_into_transformer = False

    def forward_inference_mode(self, x):
        query_states = x['query_states'][:, None, :]
        zeros = x['zeros'][:, None, :]
        state_seq = torch.cat([x['context_states'], query_states], dim=1)
        action_seq = torch.cat(
            [x['context_actions'], zeros[:, :, :self.action_dim]], dim=1)
        next_state_seq = torch.cat(
            [x['context_next_states'], zeros[:, :, :self.state_dim]], dim=1)
        reward_seq = torch.cat([x['context_rewards'], zeros[:, :, :1]], dim=1)
        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        seq_len = seq.shape[1]
        stacked_inputs = self.embed_transition(seq)

        transformer_outputs = self.transformer(  # Automaticallly causal
            inputs_embeds=stacked_inputs,
            output_attentions=self.save_activations,
            output_hidden_states=self.save_activations
        )
        if self.save_activations:
            self.activations = transformer_outputs
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])
        return preds[:, -1, :]

    def forward_training_mode(self, x):
        query_states = x['query_states'][:, None, :]
        zeros = x['zeros'][:, None, :]
        seq_len = x['context_states'].shape[1]

        # Interleave query states with context states
        state_seq = []
        action_seq = []
        next_state_seq = []
        reward_seq = []
        query_locations = []
        query_every = 10 if seq_len < 1000 else 20
        for i in range(seq_len):
            state_seq.append(x['context_states'][:, i, :].unsqueeze(1))
            action_seq.append(x['context_actions'][:, i, :].unsqueeze(1))
            next_state_seq.append(x['context_next_states'][:, i, :].unsqueeze(1))
            reward_seq.append(x['context_rewards'][:, i, :].unsqueeze(1))
            query_locations.append(0)
            if (i % query_every == 0) or (i == seq_len - 1):
                state_seq.append(query_states)
                action_seq.append(zeros[:, :, :self.action_dim])
                next_state_seq.append(zeros[:, :, :self.state_dim])
                reward_seq.append(zeros[:, :, :1])
                query_locations.append(1)
        state_seq = torch.cat(state_seq, dim=1)
        action_seq = torch.cat(action_seq, dim=1)
        next_state_seq = torch.cat(next_state_seq, dim=1)
        reward_seq = torch.cat(reward_seq, dim=1)
        query_locations = torch.tensor(query_locations, dtype=torch.bool)
        stacked_seq_len = query_locations.size(0)
        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        stacked_inputs = self.embed_transition(seq)

        # Make attention matrix
        # Create attention mask matrix of query_locations size
        attention_mask = torch.tril(torch.ones(stacked_seq_len, stacked_seq_len))
        attention_mask[:, query_locations] = 0
        attention_mask[query_locations, query_locations] = 1
        attention_mask = attention_mask.bool().to(stacked_inputs.device)

        # Pass into transformer
        transformer_inputs = {
            'inputs_embeds': stacked_inputs,
            'attention_mask': attention_mask
        }
        if self.pass_query_locations_into_transformer:
            transformer_inputs['query_locations'] = query_locations
        transformer_outputs = self.transformer(**transformer_inputs)
        if torch.isnan(transformer_outputs['last_hidden_state']).any():
            print("NaNs detected in the transformer outputs forward pass")
            raise ValueError("NaNs detected in the forward pass")

        preds = self.pred_actions(transformer_outputs['last_hidden_state'])

        if torch.isnan(preds).any():
            print("NaNs detected in the pred forward pass")
            raise ValueError("NaNs detected in the forward pass")

        if self.test:
            return preds[:, -1, :]
        if self.train_on_last_pred_only:
            indices = torch.tensor([preds.shape[1] - 1]).to(preds.device)
        else:
            indices = torch.argwhere(query_locations).squeeze().to(preds.device)
        preds_subset = torch.index_select(preds, 1, indices)
        return preds_subset

    def forward(self, x):
        """ Sequences are in (batch_size, seq_len, dim) """
        if self.test:
            return self.forward_inference_mode(x)
        else:
            return self.forward_training_mode(x)

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

        if torch.isnan(loss).any():
            print("NaNs detected in the loss")
            raise ValueError("NaNs detected in the loss")
        if (loss.abs()==torch.inf).any():
            print("Infs detected in the loss")
            raise ValueError("Infs detected in the loss")

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
        optimizer_dict = {'optimizer': optimizer}

        if self.optimizer_config['use_scheduler']:
            lr_scheduler = {  # linearly decrease LR
                'scheduler': torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=50,
            ),
            'monitor': 'val_loss',
            }
            optimizer_dict['lr_scheduler'] = lr_scheduler
        return optimizer_dict
    
    def remove_unused_params(self, model):
        try:
            delattr(model, 'wpe')
        except:
            print('Attempted to delete WPE in transformer, but could not find.')
        try:
            delattr(model, 'wte')
        except:
            print('Attempted to delete WTE in transformer, but could not find.')

    #def on_after_backward(self):
    #    # Check for NaNs in gradients
    #    for name, param in self.named_parameters():
    #        if param.grad is not None and torch.isnan(param.grad).any():
    #            print(f"NaNs detected in gradients of {name}")
    #            raise ValueError(f"NaNs detected in gradients of {name}")
    #        
    #        if torch.isnan(param).any():
    #            print(f"NaNs detected in model parameters of {name}")
    #            raise ValueError(f"NaNs detected in model parameters of {name}")
    #        
    #        norm = torch.norm(param.data).item()
    #        print(f"{name}: {norm:.6f}")

class FixedMemoryTransformer(Transformer):
    """Transformer class."""

    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        n_head: int,
        state_dim: int,
        action_dim: int,
        dropout: float,
        train_on_last_pred_only: bool,
        test: bool,
        name: str,
        optimizer_config: dict,
        ):

        pl.LightningModule.__init__(self)

        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout = dropout
        self.train_on_last_pred_only = train_on_last_pred_only
        self.test = test
        self.optimizer_config = optimizer_config = optimizer_config

        config = GPT2Config(
            n_positions=2000,  # Position embeddings are not used, but make sure it's > seq length for inference mode
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = FixedMemoryGPT2Model(config)
        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim + 1, self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.save_activations = False
        self.pass_query_locations_into_transformer = True
