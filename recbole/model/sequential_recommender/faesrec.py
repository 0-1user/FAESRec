import torch
import torch.nn as nn
import random
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.data.interaction import Interaction
import torch.nn.functional as F
import math
import copy
from torch.nn.modules.module import Module
import numpy as np
from timm.models.layers import trunc_normal_

class FAESRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(FAESRec, self).__init__(config, dataset)
        
        # load parameters info
        self.dataset = dataset
        self.config = config
        self.hidden_size = config["hidden_size"]  
        self.layer_norm_eps = config["layer_norm_eps"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.loss_type = config["loss_type"]
        self.initializer_range = config["initializer_range"]
        self.temperature = config["temperature"]
        self.batch_size = config["train_batch_size"]
        self.l1_weight = config["l1_weight"]
        self.cl_weight = config["cl_weight"]

        self.sim = config["sim"]
       
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.encoder = Encoder(config)
        
        self.aug_f = FreqAug(
            self.max_seq_length, 
            config)
        
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        self.criterion = nn.CosineSimilarity(dim=1)
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
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

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, item_seq, input_tensor, item_seq_len):   
        seq_output = self.encoder(item_seq, input_tensor)  
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1) 
        return seq_output  

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        input_tensor = self.add_position_embedding(item_seq) 

        aug_seq_emb = self.aug_f(input_tensor)  

        seq_output = self.forward(item_seq, input_tensor, item_seq_len)
        aug_seq_output = self.forward(item_seq, aug_seq_emb, item_seq_len) 

        pos_items = interaction[self.POS_ITEM_ID]
        
        log, lab = self.info_nce(seq_output, aug_seq_output, self.sim)  
        cl_los = self.aug_nce_fct(log, lab)
        
        l1_weight_loss = torch.norm(self.aug_f.para[:, 0], p=1)  
        cl_loss = self.cl_weight * cl_los + l1_weight_loss * self.l1_weight / self.max_seq_length  

        
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss + cl_loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss + cl_loss
        
    def info_nce(self, z_i, z_j, sim):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == "cos":
            sim = (
                nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
                / self.temperature
            )
        elif sim == "dot":
            sim = torch.mm(z, z.T) / self.temperature

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
 
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        input_tensor = self.add_position_embedding(item_seq)  # B,L,D
        seq_output = self.forward(item_seq, input_tensor, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        input_tensor = self.add_position_embedding(item_seq)  # B,L,D
        seq_output = self.forward(item_seq, input_tensor, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores    

class ComplexReLU(nn.Module):
    def forward(self, x):
        real = torch.relu(x.real)
        imag = torch.relu(x.imag)
        return torch.complex(real, imag)
    
class ComplexDropout(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ComplexDropout, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, x):
        real = self.dropout(x.real)
        imag = self.dropout(x.imag)
        return torch.complex(real, imag)

class FAMoELayer(nn.Module):
    def __init__(self, config):
        super(FAMoELayer, self).__init__()
        self.expert_num = config["expert_num"]
        hidden_size = config["hidden_size"]
        self.hidden_size = hidden_size
        self.seq_len = 50
        self.freq_len = self.seq_len // 2 + 1
        
        #参数化频带边界
        self.band_boundaries = nn.Parameter(torch.rand(self.expert_num - 1))

        self.gating_network = nn.Sequential(
                                            nn.Linear(self.freq_len, self.freq_len), 
                                            nn.ReLU(), 
                                            nn.Linear(self.freq_len, self.expert_num))
        self.complex_weight = nn.Parameter(torch.randn(1, self.seq_len//2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

   
    def forward(self, x):  
        
        freq_x = torch.fft.rfft(x, dim=-1)  

        total_freq_size = freq_x.size(-1)   
        boundaries = torch.sigmoid(self.band_boundaries)
        boundaries, _ = torch.sort(boundaries) 

        boundaries = torch.cat([
            torch.tensor([0.0], device=boundaries.device),
            boundaries,
            torch.tensor([1.0], device=boundaries.device)
        ])

        indices = (boundaries * total_freq_size).long() #[ 0, 15, 17, 18, 26]

        indices[-1] = total_freq_size
        
        components = []
        for i in range(self.expert_num):
            start_idx = indices[i].item()   
            end_idx = indices[i + 1].item() 

            freq_mask = torch.zeros_like(freq_x)    
            if end_idx > start_idx:
                freq_mask[:, :, start_idx:end_idx] = 1

            expert_component = freq_x * freq_mask   

            components.append(expert_component.unsqueeze(-1))  

        freq_magnitude = torch.abs(freq_x)  

        gating_input = freq_magnitude.mean(dim=1)   

        gating_scores = nn.Softmax(dim=-1)(self.gating_network(gating_input))   
        
        components = torch.cat(components, dim=-1)

        gating_scores = gating_scores.unsqueeze(1).unsqueeze(2) 

        combined_freq_output = torch.sum(components * gating_scores, dim=-1)    
        
        combined_output = torch.fft.irfft(combined_freq_output, n=self.seq_len)
       
        combined_output = combined_output.permute(0, 2, 1)  
        return combined_output
    

class FreqAug(Module):
    def __init__(self, max_seq_length, config):
        super(FreqAug, self).__init__()
       
        self.weight = nn.Parameter(torch.empty((max_seq_length//2 + 1, 2)))
        self.reset_parameters()

    def get_sampling(self, weight, temperature=0.1, bias=0.0):

        if self.training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(weight.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.cuda()
            gate_inputs = (gate_inputs + weight) #/ temperature # todo adaptive temperature
            para = F.gumbel_softmax(gate_inputs, tau=0.1)#torch.sigmoid(gate_inputs)
            return para
        else:
            return F.gumbel_softmax(weight)#torch.sigmoid(weight)
        
    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.10)
    
    def forward(self,x):
        para = self.get_sampling(self.weight, temperature=0.1)
        self.para = para

        noise_para = self.weight.detach().clone() * (-1)
        noise_para[noise_para < max(0, noise_para[:, 0].mean())] = 0.0
        scaling_factor = 1.0 / noise_para[:, 0][noise_para[:, 0] != 0].mean()

        x_ft = torch.fft.rfft(x, dim=-2)    #B，L，D
        x_ft = x_ft * torch.unsqueeze(para[:, 0] + noise_para[:, 0]*scaling_factor, -1)
        aug = torch.fft.irfft(x_ft, n=x.shape[-2], dim=-2)
        return aug

def swish(x):
    return x * torch.sigmoid(x)
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        num_blocks = config["n_layers"]
        layer = FAEBlock(config)

        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(num_blocks)])

    def forward(self, item_seq, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        
        #all_encoder_layers = all_encoder_layers[0] + all_encoder_layers[1]  # B,L,D
        return all_encoder_layers[-1]

class Intermediate(nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["hidden_size"] * 4)
        if isinstance(config["hidden_act"], str):
            self.intermediate_act_fn = ACT2FN[config["hidden_act"]]
        else:
            self.intermediate_act_fn = config["hidden_act"]

        self.dense_2 = nn.Linear(4 * config["hidden_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FAEBlock(nn.Module):
    def __init__(self, config):
        super(FAEBlock, self).__init__()
        self.moe = FAMoELayer(config)
        self.freadaptorlayer = FreAdaptorLayer(config)
        self.enc = MambaLayer(config)
        self.intermediate = Intermediate(config)
        self.norm1 = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
    
    def forward(self, hidden_states):
        hidden_states = self.moe(hidden_states.permute(0, 2, 1))    
        hidden_states = self.freadaptorlayer(hidden_states)
        
        hidden_states = self.enc(hidden_states)
        sequence_output = self.intermediate(hidden_states)  
        return sequence_output

class LearnableFilterLayer(nn.Module):
    def __init__(self, dim):
        super(LearnableFilterLayer, self).__init__()
        self.complex_weight_1 = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_2 = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_relu = ComplexReLU()

        trunc_normal_(self.complex_weight_1, std=.02)
        trunc_normal_(self.complex_weight_2, std=.02)

    def forward(self, x_fft):
        weight_1 = torch.view_as_complex(self.complex_weight_1)
        weight_2 = torch.view_as_complex(self.complex_weight_2)
        x_weighted = x_fft * weight_1
        x_weighted = self.complex_relu(x_weighted)
        x_weighted = x_weighted * weight_2
        return x_weighted
             
class FreAdaptorLayer(nn.Module):
    def __init__(self, config):
        super(FreAdaptorLayer, self).__init__()
        dim = config["hidden_size"]
        self.adaptive_filter = True
        self.learnable_filter_layer_1 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_2 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_3 = LearnableFilterLayer(dim)

        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
        self.low_pass_cut_freq_param = nn.Parameter(dim // 2 - torch.rand(1) * 0.5)
        self.high_pass_cut_freq_param = nn.Parameter(dim // 4 - torch.rand(1) * 0.5)
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)
        self.ac = nn.SiLU()

        self.norm = nn.LayerNorm(dim, eps=1e-12)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
    
    def adaptive_freq_pass(self, x_fft, flag="high"): 
        B, H, W_half = x_fft.shape  
        W = (W_half - 1) * 2 
        
       
        freq = torch.fft.rfftfreq(W, d=1/W).to(x_fft.device)
        
        if flag == "high": 
            freq_mask = torch.abs(freq) >= self.high_pass_cut_freq_param.to(x_fft.device)
        else:
            freq_mask = torch.abs(freq) <= self.low_pass_cut_freq_param.to(x_fft.device)
        return x_fft * freq_mask

    def forward(self, x_in):    
        B, N, C = x_in.shape

        x = x_in.to(torch.float32)
        
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')   

        if self.adaptive_filter:
            
            x_low_pass = self.adaptive_freq_pass(x_fft, flag="low")
            
            x_high_pass = self.adaptive_freq_pass(x_fft, flag="high")

        x_weighted = self.learnable_filter_layer_1(x_fft) + self.learnable_filter_layer_3(x_high_pass) + self.learnable_filter_layer_2(x_low_pass) 
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')  
        
        return x
       
from mamba_ssm import Mamba
class MambaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config["n_layers"]
        self.mamba = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=config["hidden_size"],
                d_state=config["d_state"],
                d_conv=config["d_conv"],
                expand=config["expand"],
            )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=1e-12)

    def forward(self, input_tensor):
       
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:        # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:                           # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        return hidden_states
