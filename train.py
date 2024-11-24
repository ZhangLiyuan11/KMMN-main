from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import logging, BertConfig, BertModel
from torch.autograd import Variable
from weibo_dataset import *

from network import MutiModel
import numpy as np
import torch.nn as nn


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def prepre():
    batch_size = 64
    lr_clip = 0.0005
    l2 = 0
    alpha = 0.4
    beta = 0.1
    # Set CUDA device if available
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create dataset
    train_set = weibo_dataset(is_train=True)
    test_set = weibo_dataset(is_train=False)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=12, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=12, collate_fn=collate_fn, shuffle=False)

    # Initialize the MultiModal network
    rumor_model = MutiModel().cuda()

    # Define the optimizer
    optim_rumor = torch.optim.Adam(
        rumor_model.parameters(), lr=lr_clip, weight_decay=l2)

    # Define the loss function for rumor classification
    loss_f_rumor = torch.nn.CrossEntropyLoss().cuda()
    
    loss_f_bce = nn.BCEWithLogitsLoss().cuda()
    clipmodel,clip_cn_preprocess = load_chinese_clip_model(device)
    return (alpha,beta,device, train_loader, test_loader, rumor_model,optim_rumor,loss_f_rumor,loss_f_bce,clipmodel)


def train():
    (alpha,beta,device, train_loader, test_loader, rumor_model,optim_rumor,loss_f_rumor,loss_f_bce,clipmodel) = prepre()
    for epoch in range(25):
        rumor_model.train()
        corrects_pre_rumor = 0
        loss_rumor_total = 0
        loss_coarse_grained = 0
        loss_fine_frained = 0
        rumor_count = 0
        loss_final_total = 0
        for i, (images_swin, image_clip,text,caption,text_bert,content_entities_k,image_entities_k,new_entities_k,labels) in tqdm(enumerate(train_loader)):
            image_clip_pre, image_swin, labels = torch.squeeze(to_var(image_clip)), to_var(images_swin),to_var(labels)
            # content_entities_k, image_entities_k,new_entities_k = to_var(content_entities_k), to_var(image_entities_k),to_var(new_entities_k)
            text_pre = clip_cn.tokenize(text).to(device)
            with torch.no_grad():
                image_clip = clipmodel.encode_image(image_clip_pre)
                text_clip = clipmodel.encode_text(text_pre)
            project_sim,pre_final,clip_signal_score,prime_signal_score = rumor_model(image_clip,text_clip,text,text_bert,image_swin,caption,content_entities_k,image_entities_k,new_entities_k)

            loss_final = loss_f_rumor(pre_final,labels)
            # loss_coarse = loss_f_bce(torch.squeeze(clip_signal_score),labels.float())
            # loss_fine = loss_f_bce(torch.squeeze(prime_signal_score),labels.float())
            loss_coarse = loss_f_rumor(clip_signal_score,labels)
            loss_fine = loss_f_rumor(prime_signal_score,labels)
            loss_sim = loss_f_rumor(project_sim,labels)
            loss_rumor = loss_final + alpha * (loss_coarse + loss_fine) + beta * loss_sim

            # torch.nn.utils.clip_grad_value_(rumor_module.parameters(), clip_value=0.5)
            optim_rumor.zero_grad()
            loss_rumor.backward()
            optim_rumor.step()

            pre_label_rumor = pre_final.argmax(1)
            corrects_pre_rumor += pre_label_rumor.eq(labels.view_as(pre_label_rumor)).sum().item()

            batch_size = image_clip.shape[0]
            loss_final_total += loss_final.item() * batch_size
            loss_rumor_total += loss_rumor.item() * batch_size
            loss_coarse_grained += loss_coarse.item() * batch_size
            loss_fine_frained += loss_fine.item() * batch_size
            rumor_count += batch_size

        loss_rumor_train_cross = loss_rumor_total / rumor_count
        loss_final_train = loss_final_total / rumor_count
        loss_corase_train = loss_coarse_grained / rumor_count
        loss_frain_train = loss_fine_frained / rumor_count

        acc_rumor_train = corrects_pre_rumor / rumor_count

        acc_rumor_test, loss_final_test,loss_fine_frained_test,loss_coarse_grained_test, conf_rumor = test(rumor_model, test_loader,clipmodel)
        print('-----------rumor detection----------------')
        print(
            "EPOCH = %d || acc_rumor_train = %.3f || acc_rumor_test = %.3f || loss_rumor_train = %.3f || "
            "loss_final_train = %.3f || loss_final_test = %.3f || loss_train_coarse = %.3f || loss_train_frain = %.3f ||"
            "loss_fine_frained_test = %.3f || loss_coarse_grained_test = %.3f" %
            (epoch + 1, acc_rumor_train, acc_rumor_test,loss_rumor_train_cross,loss_final_train,
             loss_final_test,loss_corase_train,loss_frain_train,loss_fine_frained_test,loss_coarse_grained_test))
        print('-----------rumor_confusion_matrix---------')
        print(conf_rumor)


def test(rumor_model, test_loader,clipmodel):
    rumor_model.eval()
    loss_f_rumor = torch.nn.CrossEntropyLoss().cuda()
    loss_f_bce = nn.BCEWithLogitsLoss().cuda()
    loss_final_total = 0
    loss_coarse_grained_total = 0
    rumor_count = 0
    loss_fine_frained_total = 0
    rumor_label_all = []
    rumor_pre_label_all = []
    with (torch.no_grad()):
        for i, (images_swin, image_clip,text,caption,text_bert,content_entities_k,image_entities_k,new_entities_k,labels) in tqdm(enumerate(test_loader)):
            # content_entities_k, image_entities_k,new_entities_k = to_var(content_entities_k), to_var(image_entities_k),to_var(new_entities_k)
            image_clip_pre, image_swin,labels = torch.squeeze(to_var(image_clip)), to_var(images_swin),to_var(labels)
            text_pre = clip_cn.tokenize(text).to(device)
            with torch.no_grad():
                image_clip = clipmodel.encode_image(image_clip_pre)
                text_clip = clipmodel.encode_text(text_pre)
            project_sim,pre_final,clip_signal_score,prime_signal_score = rumor_model(image_clip,text_clip,text,text_bert,image_swin,caption,content_entities_k,image_entities_k,new_entities_k)

            # loss_coarse_grained_test = loss_f_bce(torch.squeeze(clip_signal_score), labels.float())
            # loss_fine_frained_test = loss_f_bce(torch.squeeze(prime_signal_score), labels.float())
            loss_coarse_grained_test = loss_f_rumor(clip_signal_score,labels)
            loss_fine_frained_test = loss_f_rumor(prime_signal_score,labels)
            loss_final = loss_f_rumor(pre_final, labels)

            batch_size = image_clip.shape[0]
            loss_final_total += loss_final.item() * batch_size
            loss_coarse_grained_total += loss_coarse_grained_test.item() * batch_size
            loss_fine_frained_total += loss_fine_frained_test.item() * batch_size
            rumor_count += batch_size

            # Store predicted and true labels for evaluation
            pre_label_rumor = pre_final.argmax(1)
            rumor_pre_label_all.append(pre_label_rumor.detach().cpu().numpy())
            rumor_label_all.append(labels.detach().cpu().numpy())

        # Calculate accuracy and confusion matrix
        loss_final_test = loss_final_total / rumor_count
        loss_prime_test = loss_fine_frained_total / rumor_count
        loss_clip_test = loss_coarse_grained_total / rumor_count

        rumor_pre_label_all = np.concatenate(rumor_pre_label_all, 0)
        rumor_label_all = np.concatenate(rumor_label_all, 0)
        acc_rumor_test = accuracy_score(rumor_pre_label_all, rumor_label_all)
        conf_rumor = confusion_matrix(rumor_pre_label_all, rumor_label_all)

    return acc_rumor_test, loss_final_test,loss_prime_test,loss_clip_test, conf_rumor


train()
