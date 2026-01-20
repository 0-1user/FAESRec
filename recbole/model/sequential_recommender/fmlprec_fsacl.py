import torch
import torch.nn as nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.data.interaction import Interaction
import torch.nn.functional as F
import math
import copy

class FMLPRec_FSaCL(SequentialRecommender):
    def __init__(self, config, dataset):
        super(FMLPRec_FSaCL, self).__init__(config, dataset)
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
        
        self.l1_weight = config["l1_weight"]
        self.cl_weight = config["cl_weight"]
        self.sim = config["sim"]
        self.temperature = config["temperature"]
        self.batch_size = config["train_batch_size"]
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.aug_f = FreqAug(
            self.max_seq_length, 
            config)
        self.criterion = nn.CosineSimilarity(dim=1)
        self.aug_nce_fct = nn.CrossEntropyLoss()
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
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
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
    # same as SASRec
    def forward(self, input_tensor, item_seq_len):

        

        item_encoded_layers = self.item_encoder(input_tensor,
                                                output_all_encoded_layers=True,
                                                )
        output = item_encoded_layers[-1]
        
        output = self.gather_indexes(output, item_seq_len - 1)
        return output
    
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        input_tensor = self.add_position_embedding(item_seq)
        seq_output = self.forward(input_tensor, item_seq_len)

        aug_seq_emb = self.aug_f(input_tensor)  # B,L,D 频域增强
        aug_seq_output = self.forward(aug_seq_emb, item_seq_len)  # B,L,D
        # 对比损失
        log, lab = self.info_nce(seq_output, aug_seq_output, self.sim)  # B,D
        cl_los = self.aug_nce_fct(log, lab)
        l1_weight_loss = torch.norm(self.aug_f.para[:, 0], p=1)  # L1正则化
        cl_loss = self.cl_weight * cl_los + l1_weight_loss * self.l1_weight / self.max_seq_length  # L1正则化系数
        
        pos_items = interaction[self.POS_ITEM_ID]
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

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        input_tensor = self.add_position_embedding(item_seq)  # B,L,D
        seq_output = self.forward(input_tensor, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        input_tensor = self.add_position_embedding(item_seq)  # B,L,D
        seq_output = self.forward(input_tensor, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

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

    
#频率处理
class FreqAug(nn.Module):
    def __init__(self, max_seq_length, config):
        super(FreqAug, self).__init__()
        #自适应全局语义视图
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