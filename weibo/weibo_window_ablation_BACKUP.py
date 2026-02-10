from cgitb import text
from email.mime import image
from pydoc import cli
from string import digits
from mymodel import Multi_Model
# import matplotlib.pyplot as plt
import pickle
from PIL import Image
import re
import os
import copy
from torchvision.transforms.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import warnings
from torch import LongTensor
import torch
import time
from sklearn.metrics import accuracy_score, classification_report, recall_score
# from clip import CLIP
import math

warnings.filterwarnings('ignore')

seed = 3407
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Config():
    def __init__(self):
        self.batch_size = 4
        self.epochs = 5  # 建议 >1，哪怕 3
        self.bert_path = "bert-base-chinese"
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.lr = 2e-5
        self.l2 = 1e-5

        # ⭐ 新增
        self.window_size = 2


class FakeNewsDataset(Dataset):
    def __init__(self, input_three, event, image, label):
        self.event = LongTensor(list(event))
        self.image = torch.FloatTensor([np.array(i) for i in image])
        self.label = LongTensor(list(label))
        self.input_three = list()
        self.input_three.append(LongTensor(input_three[0]))
        self.input_three.append(LongTensor(input_three[1]))
        self.input_three.append(LongTensor(input_three[2]))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.input_three[0][idx], self.input_three[1][idx], self.input_three[2][idx], self.image[idx], \
            self.event[idx], self.label[idx]


def cleanSST(string):
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)


'''def evaluate(clip_module, multi_model, vali_dataloader, device):
    clip_module.eval()
    multi_model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for index, (batch_text0, batch_text1, batch_text2, batch_image, batch_event, batch_label) in enumerate(
                vali_dataloader):
            batch_text0 = batch_text0.to(device)
            batch_text1 = batch_text1.to(device)
            batch_text2 = batch_text2.to(device)
            batch_image = batch_image.to(device)
            batch_event = batch_event.to(device)
            batch_label = batch_label.to(device)
            #image_aligned, text_aligned = clip_module(batch_text0, batch_text1, batch_text2, batch_image)  # N* 64
            #outputs = multi_model(batch_text0, batch_text1, batch_text2, batch_image, text_aligned, image_aligned)
            #y_pred = outputs[0]
            #a_s = outputs[1]

            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(batch_label.squeeze().cpu().numpy().tolist())
    print(classification_report(val_true, val_pred, target_names=['Real News', 'Fake News'], digits=3))
    return accuracy_score(val_true, val_pred)
'''


def train_val_test(window_size):
    config = Config()

    train_dataset = pickle.load(open('./pickles/new_train_dataset.pkl', 'rb'))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = Multi_Model(
        bert_path=config.bert_path,
        window_size=window_size
    )

    model.to(config.device)

    criterion = nn.CrossEntropyLoss()

    print("Start Stage 1 sanity check...")

    for batch in train_loader:
        batch_text0, _, _, _, _, batch_label = batch

        input_ids = batch_text0.to(config.device)
        attention_mask = (input_ids != 0).long().to(config.device)
        labels = batch_label.to(config.device)

        logits = model(input_ids, attention_mask)

        # ⭐ 关键补充：真正计算 loss
        loss = criterion(logits, labels)

        loss_value = loss.item()
        print(f"[RESULT] window_size={window_size}, loss={loss_value:.4f}")

        break  # ⭐ 只跑一个 batch（消融阶段推荐）

    print("Stage 1 forward + loss DONE.")


'''def prepare_data(batch_text0, batch_text1, batch_text2, batch_image, batch_label):


    # 文本保持不变（正负样本共用）
    fixed_text0 = batch_text0
    fixed_text1 = batch_text1
    fixed_text2 = batch_text2

    # 正样本：原始 image
    matched_image = batch_image

    # 负样本：打乱 batch 维度，构造不匹配图像
    batch_size = batch_image.size(0)
    perm = torch.randperm(batch_size)

    # 避免 identity permutation（极小概率，但稳妥起见）
    if torch.all(perm == torch.arange(batch_size, device=batch_image.device)):
        perm = torch.roll(perm, shifts=1)

    unmatched_image = batch_image[perm]

    return fixed_text0, fixed_text1, fixed_text2, matched_image, unmatched_image
'''

'''def soft_loss(input, target):
    logprobs = torch.nn.functional.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]
'''

train_acc_vector = []
vali_acc_vector = []

if __name__ == "__main__":
    window_sizes = [0, 1, 2, 3, 4]

    for w in window_sizes:
        print("=" * 50)
        print(f"Running experiment with window_size = {w}")
        print("=" * 50)

        train_val_test(window_size=w)
