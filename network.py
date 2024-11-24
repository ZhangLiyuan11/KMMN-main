import random

import torch
import torch.backends.cudnn as cudnn
from KM_module import *
from KCT_module import *
from tools import *
from models import *

manualseed = 64
random.seed(manualseed)
np.random.seed(manualseed)
torch.manual_seed(manualseed)
torch.cuda.manual_seed(manualseed)
cudnn.deterministic = True

class MutiModel(nn.Module):
    def __init__(self,):
        super(MutiModel, self).__init__()
        self.ClipModel = ClipMultiModal().cuda()
        self.PrimeModel = PrimeMultiModal().cuda()
        self.classifier_corre_final = nn.Sequential(
            nn.Linear(64*2+32+4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
        self.KM_mdoel = KM_mdoel(model_dimension=1024, dropout_probability=0.1,log_attention_weights=False)

        self.imoe_coarse = iMOE(3, 256, 64).cuda()
        self.imoe_frain = iMOE(3, 256, 64).cuda()

        self.bert,self.tokenizer = pretrain_bert_models()

    def forward(self,imageclip,textclip,text,text_bert,image_swin,caption,content_entities_k,image_entities_k,new_entities_k):
        news_des = caption+text_bert
        # encoding
        new_em = bert_process_poolout(news_des,self.bert,self.tokenizer)
        # KnowledgeScreen
        content_res, image_res, news_res = KnowledgeScreen(new_em,content_entities_k,image_entities_k)
        
        # coarse
        cross_fea_coarse,muti_fea_coarse,simlarity,project_sim = self.ClipModel(imageclip.float(), textclip.float())
        # frain
        cross_fea_frain,muti_fea_frain,new_fea = self.PrimeModel(text,text_bert, image_swin,news_res,self.bert,self.tokenizer)
        # Knowledge mach
        k_fea = self.KM_mdoel(content_res,image_res)

        coarse_signal_fea,coarse_signal_score = self.imoe_coarse(cross_fea_coarse,muti_fea_coarse,simlarity)
        frain_signal_fea,frain_signal_score = self.imoe_frain(cross_fea_frain,muti_fea_frain,simlarity)

        final_fea = torch.cat((coarse_signal_fea,coarse_signal_score,frain_signal_fea,frain_signal_score,k_fea), dim=1)
        
        pre_final = self.classifier_corre_final(final_fea)

        return project_sim,pre_final,coarse_signal_score,frain_signal_score

class ClipMultiModal(nn.Module):
    def __init__(self,):
        super(ClipMultiModal, self).__init__()
        self.uni_repre = UnimodalDetection()
        self.uni_clip = nn.Sequential(
            nn.Linear(1024 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self,imageclip, textclip):
        # process prime
        similarity = my_cos_sim(imageclip,textclip)
        text_prime, image_prime = self.uni_repre(textclip,imageclip)
        project_sim = my_cos(text_prime, image_prime)
        muti_fea = torch.cat((text_prime, image_prime),dim=1)
        cross_fea = torch.cat((imageclip,textclip),dim=1)
        cross_fea = self.uni_clip(cross_fea)
        return cross_fea,muti_fea,similarity,project_sim

class PrimeMultiModal(nn.Module):
    def __init__(self,):
        super(PrimeMultiModal, self).__init__()
        self.swin = pretrain_swin_models()
        self.text_cnn_encoder = TextCNN(256, 1024)
        self.KCT_trans = KCT_Transformer(model_dimension=1024, number_of_heads=8, number_of_layers=1, dropout_probability=0.1,
                                 log_attention_weights=False)
        self.signal_fea_projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout()
        )
        

    def forward(self, text,text_bert, image_swin,new_entities_k,bert,tokenizer):
        bert_token_fea_ocr = bert_process(text_bert,bert,tokenizer)
        bert_token_fea = bert_process(text,bert,tokenizer)
        # textcnn
        text_raw = self.text_cnn_encoder(bert_token_fea)
        # swin
        img_token_fea = self.swin(image_swin).last_hidden_state

        text_att, image_att = self.KCT_trans(bert_token_fea_ocr,img_token_fea,new_entities_k)

        text_att = torch.sum(text_att, dim=1) / 300
        image_att = torch.sum(image_att, dim=1) / 49
        
        prime_signal_fea = self.signal_fea_projection(torch.cat((text_att, image_att), dim=1))

        return prime_signal_fea,text_raw,torch.cat((text_att, image_att), dim=1)
