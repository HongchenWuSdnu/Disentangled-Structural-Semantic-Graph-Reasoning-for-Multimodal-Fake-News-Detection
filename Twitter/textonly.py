import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


TRAIN_TSV = 'multimodal_train_cleaned_subset.tsv'
VAL_TSV = 'multimodal_validate_cleaned_subset.tsv'
TEST_TSV = 'multimodal_test_public_cleaned_subset.tsv'


IMG_DIR = 'fakeddit_images_small'


BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 5e-5
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BERT_VERSION = 'bert-base-uncased'



class FakedditDataset(Dataset):
    def __init__(self, tsv_path, img_dir, tokenizer, max_len=128):
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Fakeddit 2_way_label: 1=Fake, 0=Real
        self.label_col = '2_way_label'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['clean_title'])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label
        }


class TextOnlyBERT(nn.Module):
    def __init__(self, bert_model_name, num_classes=2):
        super(TextOnlyBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)


        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)



def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(losses)


def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')

    return acc, f1, sum(losses) / len(losses)


# --- 4. 主程序 ---
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")


    if not os.path.exists(TRAIN_TSV) or not os.path.exists(TEST_TSV):
        print("错误：找不到TSV文件，请检查路径！")
        exit()


    tokenizer = BertTokenizer.from_pretrained(BERT_VERSION)

    print("Loading datasets...")
    train_dataset = FakedditDataset(TRAIN_TSV, IMG_DIR, tokenizer, MAX_LEN)
    val_dataset = FakedditDataset(VAL_TSV, IMG_DIR, tokenizer, MAX_LEN)
    test_dataset = FakedditDataset(TEST_TSV, IMG_DIR, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)

    model = TextOnlyBERT(BERT_VERSION)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)


    best_f1 = 0

    print("Start Training...")
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE, scheduler)
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_f1, val_loss = eval_model(model, val_loader, loss_fn, DEVICE)
        print(f'Val   loss {val_loss} accuracy {val_acc} F1 {val_f1}')


        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model_fakeddit_textonly.pth')
            print("=> Best model saved!")


    print("\nTraining Complete. Testing best model...")
    model.load_state_dict(torch.load('best_model_fakeddit_textonly.pth'))
    test_acc, test_f1, test_loss = eval_model(model, test_loader, loss_fn, DEVICE)

    print(f"\nFinal Test Results on Fakeddit:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")