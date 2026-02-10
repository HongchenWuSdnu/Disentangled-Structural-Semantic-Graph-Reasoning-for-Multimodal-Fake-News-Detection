import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from torch.utils.data import Dataset
import os


TEST_TSV = 'multimodal_test_public_cleaned_subset.tsv'
IMG_DIR = 'fakeddit_images_small'
BATCH_SIZE = 32
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BERT_VERSION = 'bert-base-uncased'
MODEL_PATH = 'best_model_fakeddit_textonly.pth'  # 刚才训练保存的文件


class FakedditDataset(Dataset):
    def __init__(self, tsv_path, img_dir, tokenizer, max_len=128):
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_col = '2_way_label'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['clean_title'])
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(),
                'label': label}



class TextOnlyBERT(nn.Module):
    def __init__(self, bert_model_name, num_classes=2):
        super(TextOnlyBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)


if __name__ == '__main__':
    print(f"Loading best model from {MODEL_PATH} ...")

    tokenizer = BertTokenizer.from_pretrained(BERT_VERSION)
    test_dataset = FakedditDataset(TEST_TSV, IMG_DIR, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)  # num_workers改小点省点力

    model = TextOnlyBERT(BERT_VERSION)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_targets = []

    print("Start Testing...")
    with torch.no_grad():
        for d in tqdm(test_loader):
            input_ids = d["input_ids"].to(DEVICE)
            attention_mask = d["attention_mask"].to(DEVICE)
            targets = d["label"].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    p = precision_score(all_targets, all_preds, average='macro')
    r = recall_score(all_targets, all_preds, average='macro')

    print("\n" + "=" * 30)
    print("FINAL FAKEDDIT TEXT-ONLY RESULTS")
    print("=" * 30)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 30)