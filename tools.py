import torch
from transformers import BertTokenizer, SwinModel
from transformers import  BertConfig, BertModel
from modules import *

def my_cos(representation1, representation2):
    sim = (1 + ((torch.sum(representation1 * representation2, 1) / (
            torch.sqrt(torch.sum(torch.pow(representation1, 2), 1)) * torch.sqrt(
        torch.sum(torch.pow(representation2, 2), 1))) + 1e-8))) / 2
    distance = torch.ones_like(sim) - sim
    # 前面一个应该对应的是0的预测结果，也就是谣言
    sim = torch.stack((distance, sim), dim=1)
    return sim


def pretrain_bert_models():
    tokenizer = BertTokenizer.from_pretrained("pretrain_models/roberta_wwm")
    model = BertModel.from_pretrained("pretrain_models/roberta_wwm").cuda()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def pretrain_swin_models():
    swin = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224").cuda()
    for param in swin.parameters():
        param.requires_grad = False
    return swin


def bert_process(txt, model, token):
    data = token.batch_encode_plus(batch_text_or_text_pairs=txt, truncation=True, padding='max_length', max_length=300,
                                   return_tensors='pt', return_length=True)

    # Prepare input data for the model
    input_ids = data['input_ids'].cuda()
    attention_mask = data['attention_mask'].cuda()
    token_type_ids = data['token_type_ids'].cuda()

    BERT_feature = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)

    last_hidden_states = BERT_feature['last_hidden_state']

    return last_hidden_states.cuda()


def bert_process_poolout(txt, model, token):
    data = token.batch_encode_plus(batch_text_or_text_pairs=txt, truncation=True, padding='max_length', max_length=300,
                                   return_tensors='pt', return_length=True)

    # Prepare input data for the model
    input_ids = data['input_ids'].cuda()
    attention_mask = data['attention_mask'].cuda()
    token_type_ids = data['token_type_ids'].cuda()

    BERT_feature = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)

    bert_fea = BERT_feature['pooler_output']

    return bert_fea.cuda()


def get_mask_matrix(news_em, entitie_T_fea, entities_V_fea):
    NT_dot_matrix = torch.matmul(news_em.unsqueeze(1), entitie_T_fea.transpose(1, 2))
    # NT_cos_matrix = F.cosine_similarity(news_em, entitie_T_fea, dim=1)
    NT_soft_matrix = F.softmax(torch.squeeze(NT_dot_matrix), dim=1)
    threshold_NT = NT_soft_matrix.mean(dim=(1)).unsqueeze(1) / 2

    NT_mask_matrix = torch.where(NT_soft_matrix < threshold_NT, torch.tensor(0.0), NT_soft_matrix)
    # Softmax
    NT_enhance_matrix = F.softmax(NT_mask_matrix, dim=1)
    fea_te = torch.matmul(NT_enhance_matrix.unsqueeze(1), entitie_T_fea)

    NV_dot_matrix = torch.matmul(news_em.unsqueeze(1), entities_V_fea.transpose(1, 2))
    # NV_cos_matrix = F.cosine_similarity(news_em, entities_V_fea, dim=1)
    NV_soft_matrix = F.softmax(torch.squeeze(NV_dot_matrix), dim=1)
    threshold_NV = NV_soft_matrix.mean(dim=(1)).unsqueeze(1) / 2
    NV_mask_matrix = torch.where(NV_soft_matrix < threshold_NV, torch.tensor(0.0), NV_soft_matrix)
    NV_enhance_matrix = F.softmax(NV_mask_matrix, dim=1)
    fea_ve = torch.matmul(NV_enhance_matrix.unsqueeze(1), entities_V_fea)

    # mask
    co_mask_matrices = NT_mask_matrix.unsqueeze(-1) * NV_mask_matrix.unsqueeze(-2)
    co_mask_matrices = torch.where(co_mask_matrices != 0, 1, co_mask_matrices)

    return fea_te, fea_ve, co_mask_matrices
