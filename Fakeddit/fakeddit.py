import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from mymodel import Multi_Model
import time
import argparse  # 新增
import random  # 新增


# ================= 配置区 =================
class Config():
    def __init__(self):
        # ⚠️ 确保这里指向你刚刚生成的 v2 大数据
        self.train_tsv = 'fakeddit_train_v2.tsv'
        self.val_tsv = 'fakeddit_val_v2.tsv'
        self.test_tsv = 'fakeddit_test_v2.tsv'

        # 图片路径保持不变
        self.img_dir = 'fakeddit_images_small'

        self.bert_path = 'bert-base-uncased'

        # 4090D 显存够大，32 是安全的
        self.batch_size = 32

        # 大数据量下，15个Epoch足够收敛
        self.epochs = 15

        self.lr = 2e-5
        self.window_size = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= Dataset 定义 =================
class FakedditDataset(Dataset):
    def __init__(self, tsv_path, img_dir, bert_path, max_len=128):
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.img_dir = img_dir
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_len = max_len
        self.label_col = '2_way_label'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 文本
        text = str(row['clean_title'])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # 图片
        img_id = str(row['id'])
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        try:
            image = Image.open(img_path).convert("RGB").resize((224, 224))
            arr = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(arr).permute(2, 0, 1)
        except Exception:
            image_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)

        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return input_ids, attention_mask, image_tensor, label


# ================= 核心函数 =================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="  Training", leave=False):
        input_ids, attention_mask, images, labels = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, images)

        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, desc="Evaluating"):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  {desc}", leave=False):
            input_ids, attention_mask, images, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask, images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    acc = accuracy_score(labels_all, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds, average="macro")
    return acc, precision, recall, f1


def run_experiment(cfg, train_loader, val_loader, test_loader, variant_name, use_sem, use_syn, seed):

    print(f"\n{'=' * 20} Start Experiment: {variant_name} (Seed: {seed}) {'=' * 20}")


    model = Multi_Model(
        bert_path=cfg.bert_path,
        window_size=cfg.window_size,
        use_window_graph=False,
        use_semantic=use_sem,
        use_syntax=use_syn
    ).to(cfg.device)

    bert_params_ids = list(map(id, model.bert.parameters()))
    base_params = filter(lambda p: id(p) not in bert_params_ids, model.parameters())

    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': 2e-5},
        {'params': base_params, 'lr': 2e-4}
    ])

    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    save_path = f"best_model_fakeddit_{variant_name.replace(' ', '_')}_seed{seed}.pth"


    for epoch in range(cfg.epochs):
        print(f"[Epoch {epoch + 1}/{cfg.epochs}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        acc, p, r, f1 = evaluate(model, val_loader, cfg.device, desc="Validating")

        print(f"  Train Loss: {train_loss:.4f} | Val F1: {f1:.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"  => New Best Model Saved! (F1: {best_val_f1:.4f})")


    print(f"\nRunning Final Test for {variant_name}...")
    model.load_state_dict(torch.load(save_path))
    test_acc, test_p, test_r, test_f1 = evaluate(model, test_loader, cfg.device, desc="Testing")

    print(f"Final Test Result ({variant_name} Seed {seed}): Acc={test_acc:.4f}, F1={test_f1:.4f}")

    return {
        "Model": f"{variant_name} (Seed {seed})",
        "Acc": test_acc,
        "P": test_p,
        "R": test_r,
        "F1": test_f1
    }



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training')
    args = parser.parse_args()
    current_seed = args.seed


    print(f" SETTING RANDOM SEED: {current_seed} ")
    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    torch.cuda.manual_seed_all(current_seed)
    torch.backends.cudnn.deterministic = True

    cfg = Config()
    print(f"Using Device: {cfg.device}")


    print("Loading datasets (v2 Big Data)...")
    train_ds = FakedditDataset(cfg.train_tsv, cfg.img_dir, cfg.bert_path)
    val_ds = FakedditDataset(cfg.val_tsv, cfg.img_dir, cfg.bert_path)
    test_ds = FakedditDataset(cfg.test_tsv, cfg.img_dir, cfg.bert_path)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)


    experiments = [
        ("Full Model", True, True),
        # ("w_o Syntax", True, False),
        # ("w_o Semantic", False, True),
    ]

    results = []

    for name, sem, syn in experiments:

        res = run_experiment(cfg, train_loader, val_loader, test_loader, name, sem, syn, current_seed)
        results.append(res)

    print("\n" + "=" * 50)
    print("FINAL RESULTS (Test Set)")
    print("=" * 50)
    print(f"{'Model':<25} | {'Acc':<8} | {'P':<8} | {'R':<8} | {'F1':<8}")
    print("-" * 75)

    for r in results:
        print(f"{r['Model']:<25} | {r['Acc']:.4f}   | {r['P']:.4f}   | {r['R']:.4f}   | {r['F1']:.4f}")

    print("-" * 75)
    print("Done!")