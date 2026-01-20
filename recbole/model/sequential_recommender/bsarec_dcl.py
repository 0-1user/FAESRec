"""
    [Paper]
    Author: Shin, Yehjin, et al. 
    Title: "An attentive inductive bias for sequential recommendation beyond the self-attention." 
    Conference: AAAI 2024

    [Code Reference]
    https://github.com/yehjin-shin/BSARec/blob/main/src/model/sasrec.py
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import MultiHeadAttention, FeedForward
from recbole.model.loss import BPRLoss
from recbole.data.interaction import Interaction
import math
import copy
import random

class FrequencyLayer(nn.Module):
    def __init__(self, config,hidden_dropout_prob, hidden_size, c):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.c = c // 2 + 1
        self.beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor,nfft_mask):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape

        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:].clone()
        low_pass[:, self.c:, :] = 0

        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')    

        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass  + (self.beta**2) * high_pass
        

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    
class BSARecLayer(nn.Module):
    def __init__(self, 
                 config,
                 n_heads, 
                 hidden_size, 
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 layer_norm_eps,
                 alpha,
                 c
                 ):
        super(BSARecLayer, self).__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.layer_norm_eps =layer_norm_eps
        self.filter_layer = FrequencyLayer(config,hidden_dropout_prob, hidden_size, c)
        self.attention_layer = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.alpha = alpha 

    def forward(self, input_tensor, attention_mask, item_seq_len, timestamp, nfft_mask):
        
        gsp = self.attention_layer(input_tensor, attention_mask)
        dsp = self.filter_layer(input_tensor,nfft_mask)
        hidden_states = self.alpha * dsp + ( 1 - self.alpha ) * gsp
        return hidden_states
    
class BSARecBlock(nn.Module):
    def __init__(self, 
                 config,
                 n_heads, 
                 hidden_size, 
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 layer_norm_eps,
                 alpha,
                 c,
                 hidden_act,
                 intermediate_size):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(config, n_heads, hidden_size, 
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 layer_norm_eps,
                 alpha,
                 c)
        self.feed_forward = FeedForward(hidden_size, intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps)

    def forward(self, hidden_states, attention_mask, item_seq_len, timestamp, nfft_mask):
        layer_output = self.layer(hidden_states, attention_mask, item_seq_len, timestamp, nfft_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output
    
class BSARecEncoder(nn.Module):
    def __init__(self, 
                 config,
                 n_layers,
                 n_heads, 
                 hidden_size, 
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 layer_norm_eps,
                 alpha,
                 c,
                 hidden_act,
                 intermediate_size):
        super(BSARecEncoder, self).__init__()
        block = BSARecBlock(
                 config,
                 n_heads, 
                 hidden_size, 
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 layer_norm_eps,
                 alpha,
                 c,
                 hidden_act,
                 intermediate_size)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, item_seq_len,ts,mask ,output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask, item_seq_len,ts,mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers
    
class BSARec_DCL(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(BSARec_DCL, self).__init__(config, dataset)
        self.dataset = dataset
        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.c = config['c']
        self.alpha = config['alpha']
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.batch_size = config['train_batch_size']
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = BSARecEncoder(
            config = config,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            intermediate_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            c = self.c,
            alpha = self.alpha
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.lmd = config["lmd"]
        self.lmd_sem = config["lmd_sem"]
        self.ssl = config['contrast']
        self.tau = config['tau']
        self.sim = config['sim']
        self.batch_size = config['train_batch_size']
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.same_item_index = self.get_same_item_index(dataset)
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.sem_aug_nce_fct = nn.CrossEntropyLoss()
        # parameters initialization
        self.apply(self._init_weights)

    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embedding(sequence)
        position_embeddings = self.position_embedding(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb
    def forward(self, item_seq, item_seq_len, item_seq_ts):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        nfft_mask = torch.where(item_seq_ts==0, 0, 1)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, item_seq_len, item_seq_ts, nfft_mask, output_all_encoded_layers=True
        )
        
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output 
    def truncated_normal_(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor
    def get_same_item_index(self, dataset):
        same_target_index = {}
        target_item = dataset.inter_feat[self.ITEM_ID].numpy()

        for index, item_id in enumerate(target_item):
            all_index_same_id = np.where(target_item == item_id)[0]
            same_target_index[item_id] = all_index_same_id

        return same_target_index   


    def calculate_loss(self, interaction):
        same_item_index = self.same_item_index
        sem_pos_lengths = []
        sem_pos_seqs = []
        dataset = self.dataset
        target_items = interaction[self.ITEM_ID]
        for i, item_id in enumerate(target_items):
            item_id = item_id.item()
            targets_index = same_item_index[item_id]
            lens = len(targets_index)
            if lens == 0:
                print("error")
            remaining_indices = targets_index.copy()
            while len(remaining_indices) > 0:
                sample_index = random.choice(remaining_indices)
                remaining_indices = remaining_indices[remaining_indices != sample_index]
                cur_item_list = interaction[self.ITEM_SEQ][i].to("cpu")
                sample_item_list = dataset.inter_feat[self.ITEM_SEQ][sample_index]
                are_equal = torch.equal(cur_item_list, sample_item_list)
                sample_item_length = dataset.inter_feat[self.ITEM_SEQ_LEN][sample_index]

                if not are_equal or len(remaining_indices) == 0:
                    sem_pos_lengths.append(sample_item_length)
                    sem_pos_seqs.append(sample_item_list)
                    break

        sem_pos_lengths = torch.stack(sem_pos_lengths).to(self.device)
        sem_pos_seqs = torch.stack(sem_pos_seqs).to(self.device)

        interaction.update(
            Interaction({"sem_aug": sem_pos_seqs, "sem_aug_lengths": sem_pos_lengths})
        )
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq_ts = interaction['timestamp_list']
        
        seq_output = self.forward(item_seq, item_seq_len, item_seq_ts)
        pos_items = interaction[self.POS_ITEM_ID]
        
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        
         # Unsupervised NCE
        if self.ssl in ["us", "un"]:
            aug_seq_output =  self.forward(item_seq, item_seq_len, item_seq_ts)
            nce_logits, nce_labels = self.info_nce(
                seq_output,
                aug_seq_output,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim,
            )

            loss += self.lmd * self.aug_nce_fct(nce_logits, nce_labels)

        # Supervised NCE
        if self.ssl in ["us", "su"]:
            sem_aug, sem_aug_lengths = (
                interaction["sem_aug"],
                interaction["sem_aug_lengths"],
            )
            sem_aug_seq_output = self.forward(sem_aug, sem_aug_lengths)

            sem_nce_logits, sem_nce_labels = self.info_nce(
                seq_output,
                sem_aug_seq_output,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim,
            )

            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)

        if self.ssl == "us_x":
            aug_seq_output =  self.forward(item_seq, item_seq_len, item_seq_ts)
            sem_aug, sem_aug_lengths = (
                interaction["sem_aug"],
                interaction["sem_aug_lengths"],
            )

            sem_aug_seq_output = self.forward(sem_aug, sem_aug_lengths,item_seq_ts)
            sem_nce_logits, sem_nce_labels = self.info_nce(
                aug_seq_output,
                sem_aug_seq_output,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim,
            )

            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq_ts = interaction['timestamp_list']
        seq_output = self.forward(item_seq, item_seq_len, item_seq_ts)
        
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        item_seq_ts = interaction['timestamp_list']
        seq_output = self.forward(item_seq, item_seq_len, item_seq_ts)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores
    
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0)
    
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels   