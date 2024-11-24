import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch.nn.functional as F
import copy
import torch
from cn_clip.clip import load_from_name, available_models

def my_cos_sim(representation1, representation2):
    sim = (1 + ((torch.sum(representation1 * representation2, 1) / (
                torch.sqrt(torch.sum(torch.pow(representation1, 2), 1)) * torch.sqrt(
            torch.sum(torch.pow(representation2, 2), 1))) + 1e-8))) / 2
    return sim

def load_chinese_clip_model(device):
    model, preprocess = load_from_name("ViT-H-14", device=device, download_root='pre_train_models/Chinese_CLIP/')
    model.eval()
    return model,preprocess


class UnimodalDetection(nn.Module):
    def __init__(self, shared_dim=128, prime_dim=64):
        super(UnimodalDetection, self).__init__()

        self.text_uni = nn.Sequential(
            nn.Linear(1024, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
            )

        self.image_uni = nn.Sequential(
            nn.Linear(1024, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
            )

    def forward(self, text_encoding, image_encoding):
        # text_prime = selu(self.text_uni(text_encoding))
        # image_prime = selu(self.image_uni(image_encoding))
        text_prime = self.text_uni(text_encoding)
        image_prime = self.image_uni(image_encoding)
        return text_prime, image_prime


class TextCNN(nn.Module):
    def __init__(self, fea_dim, vocab_size):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.fea_dim = fea_dim

        self.channel_in = 1
        self.filter_num = 64
        self.window_size = [2, 3, 4]

        self.textcnn = nn.ModuleList(
            [nn.Conv2d(self.channel_in, self.filter_num, (K, self.vocab_size)) for K in self.window_size])
        self.linear = nn.Sequential(torch.nn.Linear(len(self.window_size) * self.filter_num, self.fea_dim),
                                    torch.nn.ReLU())

    def forward(self, inputs):
        text = inputs.unsqueeze(1)
        text = [F.relu(conv(text)).squeeze(3) for conv in self.textcnn]
        text = [F.max_pool1d(i.squeeze(2), i.shape[-1]).squeeze(2) for i in text]
        fea_text = torch.cat(text, 1)
        fea_text = self.linear(fea_text)

        return fea_text


from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class Expert_net(nn.Module):
    def __init__(self, feature_dim, expert_dim):
        super(Expert_net, self).__init__()

        p = 0
        self.dnn_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, expert_dim),
            nn.ReLU(),
            nn.Dropout(p)
        )

    def forward(self, x):
        out = self.dnn_layer(x)
        return out


class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2


# iMOE
class iMOE(nn.Module):
    def __init__(self, experts_num, FeatureDim, ExpertOutDim):
        super(iMOE, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(FeatureDim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.project_signal_fea = nn.Sequential(
            nn.Linear(FeatureDim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout()
        )
        self.project_muti_fea = nn.Sequential(
            nn.Linear(FeatureDim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout()
        )

        self.avepooling = nn.AvgPool1d(64, stride=1)
        self.maxpooling = nn.MaxPool1d(64, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.senet = nn.Sequential(
            nn.Linear(experts_num, experts_num),
            nn.GELU(),
            nn.Linear(experts_num, experts_num),
        )

        '''Experts'''
        for i in range(experts_num):
            setattr(self, "expert_layer" + str(i + 1), Expert_net(FeatureDim * 2, ExpertOutDim).cuda())
        self.Experts = [getattr(self, "expert_layer" + str(i + 1)) for i in
                        range(experts_num)]  # Experts

        '''Shared_Gate'''
        self.gate_layer = nn.Sequential(nn.Linear(FeatureDim * 3, experts_num),
                                        nn.Softmax(dim=1)).cuda()

        # 这两个东西用来调整相似度的强度
        self.w = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))


    def forward(self, signal_em, muti_em, similarity):
        similarity = similarity.unsqueeze(1)
        sim_weight = similarity * self.w + self.b

        signal_em = self.project_signal_fea(signal_em)
        muti_em = self.project_muti_fea(muti_em)

        # reweight
        signal_em = sim_weight * signal_em

        # rough predict and concat
        temp_em = torch.cat((signal_em, muti_em), dim=1)
        pre_score = self.classifier(temp_em)

        # expert net
        Expert_Out = [expert(temp_em) for expert in self.Experts]

        Experts_Out = torch.cat(([expert[:, np.newaxis, :] for expert in Expert_Out]),
                                dim=1)  #(bs,TaskExpertNum,ExpertOutDim)
        experts_att_ave = torch.squeeze(self.avepooling(Experts_Out))
        experts_att_max = torch.squeeze(self.maxpooling(Experts_Out))
        experts_weights_ave = self.senet(experts_att_ave)
        experts_weights_max = self.senet(experts_att_max)
        experts_weights = self.sigmoid(experts_weights_ave + experts_weights_max)
        Gate_out = torch.matmul(Experts_Out.transpose(1, 2), experts_weights.unsqueeze(2)).squeeze(2)

        return Gate_out, pre_score

    
def KnowledgeScreen(news_fea, content_entities_k, image_entities_k):
    content_res = []
    image_res = []
    news_res = []
    for item_new_fea, item_content_entities_k, item_image_entities_k in zip(news_fea, content_entities_k,image_entities_k):
        item_content_entities_k = item_content_entities_k.cuda()
        item_image_entities_k = item_image_entities_k.cuda()
        item_new_fea = item_new_fea.unsqueeze(0)
        if item_content_entities_k.shape[0] == 1:
            content_strongly_related_embeddings = item_content_entities_k
        else:
            nt_dot_matrix = torch.matmul(item_new_fea, item_content_entities_k.transpose(0, 1))
            nt_soft_matrix = F.softmax(nt_dot_matrix, dim=1)
            threshold_NT = nt_soft_matrix.mean()
            nt_mask_matrix = torch.where(nt_soft_matrix < threshold_NT, torch.tensor(0.0), nt_soft_matrix)
            nonzero_indices = torch.squeeze(nt_mask_matrix).nonzero()
            content_strongly_related_embeddings = item_content_entities_k[nonzero_indices].squeeze(1)

        if item_image_entities_k.shape[0] == 1:
            image_strongly_related_embeddings = item_image_entities_k
        else:
            nv_dot_matrix = torch.matmul(item_new_fea, item_image_entities_k.transpose(1, 0))
            nv_soft_matrix = F.softmax(nv_dot_matrix, dim=1)
            threshold_NV = nv_soft_matrix.mean()
            nv_mask_matrix = torch.where(nv_soft_matrix < threshold_NV, torch.tensor(0.0), nv_soft_matrix)
            nonzero_indices = torch.squeeze(nv_mask_matrix).nonzero()
            image_strongly_related_embeddings = item_image_entities_k[nonzero_indices].squeeze(1)

        news_entities = torch.cat((content_strongly_related_embeddings, image_strongly_related_embeddings), dim=0)
        pad_num = 10 - news_entities.shape[0]
        news_entities = torch.nn.functional.pad(news_entities, (0, 0, 0, pad_num), mode='constant')

        content_res.append(content_strongly_related_embeddings)
        image_res.append(image_strongly_related_embeddings)
        news_res.append(news_entities)
    news_res = torch.stack(news_res)
    return content_res, image_res, news_res
