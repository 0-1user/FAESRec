import torch
import torch.nn as nn
import numpy as np
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.data.interaction import Interaction
import torch.nn.functional as F
import math
import copy
import random
class FMLPRec_DCL(SequentialRecommender):
    def __init__(self, config, dataset):
        super(FMLPRec_DCL, self).__init__(config, dataset)
        # load parameters info
        self.dataset = dataset
        self.config = config
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.layer_norm_eps = config["layer_norm_eps"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.loss_type = config["loss_type"]
        self.initializer_range = config["initializer_range"]
        self.num_hidden_layers = config["num_hidden_layers"]

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        self.item_encoder = Encoder(
                hidden_size=self.hidden_size, 
                 hidden_dropout_prob=self.hidden_dropout_prob, 
                 hidden_act=self.hidden_act,
                 max_seq_length=self.max_seq_length,
                 num_hidden_layers=self.num_hidden_layers
        )

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
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
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
    
    # same as SASRec
    def forward(self, item_seq, item_seq_len):

        sequence_emb = self.add_position_embedding(item_seq)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                output_all_encoded_layers=True,
                                                )
        output = item_encoded_layers[-1]
        
        output = self.gather_indexes(output, item_seq_len - 1)
        return output
    
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
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            
        
        # Unsupervised NCE
        if self.ssl in ["us", "un"]:
            aug_seq_output = self.forward(item_seq, item_seq_len)
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
            aug_seq_output = self.forward(item_seq, item_seq_len)
            sem_aug, sem_aug_lengths = (
                interaction["sem_aug"],
                interaction["sem_aug_lengths"],
            )

            sem_aug_seq_output = self.forward(sem_aug, sem_aug_lengths)
            sem_nce_logits, sem_nce_labels = self.info_nce(
                aug_seq_output,
                sem_aug_seq_output,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim,
            )

            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)

        return loss


    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
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


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
class Intermediate(nn.Module):
    def __init__(self, hidden_size=64, hidden_dropout_prob=0.5, hidden_act='gelu'):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, hidden_size * 4)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

        self.dense_2 = nn.Linear(4 * hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FilterLayer(nn.Module):
    def __init__(self, 
                 hidden_size,
                 hidden_dropout_prob,
                 max_seq_length):
        super(FilterLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.complex_weight = nn.Parameter(torch.randn(1, max_seq_length//2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)


    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        #sequence_emb_fft = torch.rfft(input_tensor, 2, onesided=False)  # [:, :, :, 0]
        #sequence_emb_fft = torch.fft(sequence_emb_fft.transpose(1, 2), 2)[:, :, :, 0].transpose(1, 2)
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Layer(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 hidden_dropout_prob, 
                 hidden_act,
                 max_seq_length):
        super(Layer, self).__init__()
       
        self.filterlayer = FilterLayer(hidden_size, hidden_dropout_prob, max_seq_length)
        self.intermediate = Intermediate(hidden_size, hidden_dropout_prob, hidden_act)

    def forward(self, hidden_states):
        
        hidden_states = self.filterlayer(hidden_states)

        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output

class Encoder(nn.Module):
    def __init__(self, 
                 hidden_size=64, 
                 hidden_dropout_prob=0.5, 
                 hidden_act='gelu',
                 max_seq_length=50,
                 num_hidden_layers=2):
        super(Encoder, self).__init__()
        layer = Layer(hidden_size, hidden_dropout_prob, hidden_act, max_seq_length)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

    
