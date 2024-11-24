import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# enhance matrix
def get_enhance_matrix(news_k,text,image):
    enhence_matrix = (text @ news_k.transpose(-2, -1)) @ (image @ news_k.transpose(-2, -1)).transpose(-2, -1)
    return enhence_matrix

class KCT_Transformer(nn.Module):
    def __init__(self, model_dimension, number_of_heads, number_of_layers, dropout_probability,
                 log_attention_weights=False):
        super().__init__()
        # All of these will get deep-copied multiple times internally
        mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha)
        self.encoder = Encoder(encoder_layer)
        self.init_params()

    def init_params(self, default_initialization=False):
        if not default_initialization:
            # model.named_parameters
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, text, image,news_k):
        src_representations_batch1 = self.encoder(text,image,news_k,0)
        src_representations_batch2 = self.encoder(text,image,news_k,1)
        return src_representations_batch1, src_representations_batch2

class FeedForward(nn.Module):
    def __init__(self, model_dimension, d_ff, dropout = 0.2):
        super().__init__()
        self.ff1 = nn.Linear(model_dimension, d_ff)
        self.ff2 = nn.Linear(d_ff, model_dimension)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self,representations_batch):
        return self.norm(representations_batch + self.ff2(self.dropout(F.relu(self.ff1(representations_batch)))))

class Encoder(nn.Module):
    def __init__(self, encoder_layer):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'
        self.encoder_layer = encoder_layer
        self.ffn_layer = FeedForward(encoder_layer.model_dimension, 2 * encoder_layer.model_dimension)
    def forward(self, src1, src2,news_k,pos):
        # Forward pass through the encoder stack
        src_representations_batch = self.encoder_layer(src1, src2,news_k,pos)
        representations = self.ffn_layer(src_representations_batch)
        return representations


class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, dropout_probability, multi_headed_attention):
        super().__init__()
        # num_of_sublayers_encoder = 2
        # self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_encoder)
        self.sublayer = SublayerLogic(model_dimension, dropout_probability)
        self.multi_headed_attention = multi_headed_attention
        self.model_dimension = model_dimension
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, srb1, srb2,news_k,pos):
        # encoder_self_attention = lambda srb1, srb2: self.multi_headed_attention(query=srb1, key=srb2, value=srb2,entities_matrix=entities_matrix)
        encoder_self_attention = lambda srb1, srb2,news_k,pos: self.multi_headed_attention(query=srb1, key=srb2, value=srb2,news_k=news_k,pos=pos)
        src_representations_batch = self.sublayer(srb1, srb2, encoder_self_attention,news_k,pos)
        return self.norm(src_representations_batch)


class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, srb1, srb2, sublayer_module,news_k,pos):
        return srb1 + self.dropout(sublayer_module(self.norm(srb1), self.norm(srb2),news_k,pos))


class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 4)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value,enhence_matrix,pos):
        scores = torch.matmul(query, key.transpose(-2, -1)) 
        # scores
        enhence_matrix = torch.unsqueeze(enhence_matrix, dim=1)
        enhence_matrix = enhence_matrix.repeat(1, self.number_of_heads, 1, 1)
        # enhanced by entities
        if pos == 0:
            scores = (scores + enhence_matrix) / 2
        else:
            scores = (scores + enhence_matrix.transpose(-2, -1)) / 2
        # softmax
        scores = scores / math.sqrt(self.head_dimension)
        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        intermediate_token_representations = torch.matmul(attention_weights, value)
        # intermediate_token_representations

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value,news_k,pos):
        batch_size = query.shape[0]
        # qkv
        query, key, value,news_k = [net(x) for net, x in zip(self.qkv_nets, (query, key, value,news_k))]

        if pos == 0:
            # news_k = self.qkv_nets[0](news_k)
            enhence_matrix = get_enhance_matrix(news_k,query,key)
        else:
            # news_k = self.qkv_nets[1](news_k)
            enhence_matrix = get_enhance_matrix(news_k,key,query)

        query = query.view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
        key = key.view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
        value = value.view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)

        # qkv
        intermediate_token_representations, attention_weights = self.attention(query, key, value,enhence_matrix,pos)

        if self.log_attention_weights:
            self.attention_weights = attention_weights
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1,
                                                                              self.number_of_heads * self.head_dimension)
        # forward
        token_representations = self.out_projection_net(reshaped)
        return token_representations

def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])

