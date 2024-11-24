import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class KM_mdoel(nn.Module):
    def __init__(self, model_dimension, dropout_probability,
                 log_attention_weights=False):
        super().__init__()
        # All of these will get deep-copied multiple times internally
        self.mha = MultiHeadedAttention(model_dimension, dropout_probability, log_attention_weights)
        self.init_params()
        self.FFN = FFN(3074,512,32)
    def init_params(self, default_initialization=False):
        if not default_initialization:
            # model.named_parameters
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, content_res,image_res):
        res = []
        for content_entities,image_entities in zip(content_res,image_res):
            quary,align_content = self.mha(content_entities,image_entities,image_entities)
            sub_fea = quary - align_content
            dot_product = torch.sum(quary * align_content, dim=1, keepdim=True)
            final_fea = torch.cat((quary,align_content,sub_fea,dot_product),dim=1)
            num_rows = final_fea.shape[0]
            ave_pool_fea = F.avg_pool1d(final_fea.transpose(1, 0), kernel_size=num_rows, stride=num_rows)
            max_pool_fea = F.max_pool1d(final_fea.transpose(1, 0), kernel_size=num_rows, stride=num_rows)
            pool_fea = torch.squeeze(torch.cat((ave_pool_fea,max_pool_fea),dim=0))
            res.append(pool_fea)
        res = self.FFN(torch.stack(res))
        return res

class FFN(nn.Module):
    def __init__(self, model_dimension, d_ff,pre_dim, dropout = 0.2):
        super().__init__()
        self.ff1 = nn.Linear(model_dimension, d_ff)
        self.ff2 = nn.Linear(d_ff, pre_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(pre_dim)

    def forward(self,representations_batch):
        return self.norm(self.ff2(self.dropout(F.relu(self.ff1(representations_batch)))))

class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dimension, dropout_probability, log_attention_weights):
        super().__init__()
        self.qkv_nets = get_clones(nn.Linear(model_dimension, 512), 3)  # identity activation hence "nets"

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)


    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(512)
        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value):
        # qkv
        query, key, value = [net(x) for net, x in zip(self.qkv_nets, (query, key, value))]
        intermediate_token_representations, attention_weights = self.attention(query, key, value)

        if self.log_attention_weights:
            self.attention_weights = attention_weights
        return query,intermediate_token_representations

def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


