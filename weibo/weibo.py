from mymodel import Multi_Model
import pickle
import re
from transformers import BertModel, BertConfig, BertTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
from torch import LongTensor
import torch
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.metrics import precision_recall_fscore_support
# weibo.py 顶部
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

warnings.filterwarnings('ignore')

seed = 3407
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)



class Config():
    def __init__(self):
        self.batch_size = 4
        self.epochs = 1  # 建议 >1，哪怕 3
        self.bert_path = "bert-base-chinese"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.lr = 2e-5
        self.l2 = 1e-5

        # ⭐ 新增
        self.window_size = 2

        # ===== Ablation switches =====
        self.use_window_graph = False  # 论文第4章是全连接图，不是 window graph
        self.use_semantic = True
        self.use_syntax = True


import os
import json
import numpy as np
from PIL import Image
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch


DATA_ROOT = "/Users/yanyuhan/Projects/weibo"

class WeiboJsonDataset(Dataset):

    def __init__(self, json_path, bert_path="bert-base-chinese", max_len=128):
        self.samples = self._load_json(json_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_len = max_len


        self.rumor_img_dir = os.path.join(DATA_ROOT, "data", "rumor_images")
        self.nonrumor_img_dir = os.path.join(DATA_ROOT, "data", "nonrumor_images")


        self._missing_img = 0
        self._printed = False


    def _load_json(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)


        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            out = []
            for k, v in data.items():
                if isinstance(v, dict):
                    vv = dict(v)
                    vv.setdefault("id", k)
                    out.append(vv)
            if len(out) > 0:
                return out

        raise ValueError(f"Unrecognized json structure: {json_path}")

    def _get_label(self, item):

        if "label" in item:
            return int(item["label"])
        if "y" in item:
            return int(item["y"])
        if "rumor" in item:
            return int(item["rumor"])
        if "is_rumor" in item:
            return 1 if item["is_rumor"] else 0

        raise ValueError(f"Cannot find label field. keys={list(item.keys())}")

    def _get_text(self, item):
        for k in ["text", "content", "tweet", "sentence"]:
            if k in item and item[k] is not None:
                return str(item[k])
        return ""

    def _get_id(self, item):

        for k in ["id", "post_id", "mid", "weibo_id"]:
            if k in item and item[k] is not None:
                return str(item[k])
        return None

    def _resolve_image_path(self, item, label):

        img_dir = self.rumor_img_dir if label == 1 else self.nonrumor_img_dir


        for k in ["image", "img", "img_path", "image_path", "image_name"]:
            if k in item and item[k]:
                name = os.path.basename(str(item[k]).strip())
                p = os.path.join(img_dir, name)
                if os.path.exists(p):
                    return p

        pid = self._get_id(item)
        if pid is not None:
            pid = str(pid).strip()
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                p = os.path.join(img_dir, pid + ext)
                if os.path.exists(p):
                    return p

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        label = self._get_label(item)
        text = self._get_text(item)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)  # [L]
        token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids)).squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)


        img_path = self._resolve_image_path(item, label)
        if img_path is None:
            self._missing_img += 1
            image = torch.zeros(3, 224, 224, dtype=torch.float32)
        else:

            img = Image.open(img_path).convert("RGB").resize((224, 224))
            arr = np.array(img).astype(np.float32) / 255.0  # [H,W,3]
            image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]

        event = torch.tensor(0, dtype=torch.long)  # 当前代码没用 event


        if (not self._printed) and idx < 3:
            self._printed = True
            print("\n[SampleCheck]")
            print("idx:", idx)
            print("label:", label)
            print("text[:80]:", text[:80])
            for k in ["image", "img", "img_path", "image_path", "image_name"]:
                if k in item:
                    print(f"raw[{k}]:", str(item[k])[:120])
                    break
            print("resolved_img_path:", img_path)
            print("img_tensor_min/max:", float(image.min()), float(image.max()))
            print("attn_valid_len:", int(attention_mask.sum().item()))


        return input_ids, token_type_ids, attention_mask, image, event, torch.tensor(label, dtype=torch.long)


import json
import os
import hashlib

def _load_json_any(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        out = []
        for k, v in data.items():
            if isinstance(v, dict):
                vv = dict(v)
                vv.setdefault("id", k)
                out.append(vv)
        if len(out) > 0:
            return out

    raise ValueError(f"Unrecognized json structure: {path}")

def _get_first(item, keys, default=None):
    for k in keys:
        if k in item and item[k] is not None and str(item[k]).strip() != "":
            return item[k]
    return default

def _norm_text(t: str) -> str:

    return " ".join(str(t).strip().lower().split())

def _hash_text(t: str) -> str:
    t = _norm_text(t)
    return hashlib.md5(t.encode("utf-8")).hexdigest()

def leakage_check(train_json, val_json, test_json, topk=10):
    train = _load_json_any(train_json)
    val   = _load_json_any(val_json)
    test  = _load_json_any(test_json)

    print("\n" + "=" * 70)
    print("[LeakCheck] Dataset sizes")
    print("=" * 70)
    print(f"train: {len(train)} | val: {len(val)} | test: {len(test)}")

    def collect(items):
        ids = set()
        imgs = set()
        txts = set()

        for it in items:
            _id = _get_first(it, ["id", "post_id", "mid", "weibo_id"])
            if _id is not None:
                ids.add(str(_id))

            img = _get_first(it, ["image", "img", "img_path", "image_path", "image_name"])
            if img is not None:
                imgs.add(os.path.basename(str(img).strip()))

            txt = _get_first(it, ["text", "content", "tweet", "sentence"], default="")
            if txt is not None and str(txt).strip() != "":
                txts.add(_hash_text(str(txt)))

        return ids, imgs, txts

    tr_ids, tr_imgs, tr_txts = collect(train)
    va_ids, va_imgs, va_txts = collect(val)
    te_ids, te_imgs, te_txts = collect(test)

    def report_overlap(a_name, a_set, b_name, b_set):
        inter = a_set & b_set
        print(f"{a_name} ∩ {b_name}: {len(inter)}")
        if len(inter) > 0:
            examples = list(inter)[:topk]
            print(f"  examples: {examples}")

    print("\n" + "=" * 70)
    print("[LeakCheck] ID overlap")
    print("=" * 70)
    report_overlap("train_id", tr_ids, "val_id", va_ids)
    report_overlap("train_id", tr_ids, "test_id", te_ids)
    report_overlap("val_id",   va_ids, "test_id", te_ids)

    print("\n" + "=" * 70)
    print("[LeakCheck] Image filename overlap")
    print("=" * 70)
    report_overlap("train_img", tr_imgs, "val_img", va_imgs)
    report_overlap("train_img", tr_imgs, "test_img", te_imgs)
    report_overlap("val_img",   va_imgs, "test_img", te_imgs)

    print("\n" + "=" * 70)
    print("[LeakCheck] Text hash overlap (same normalized text)")
    print("=" * 70)
    report_overlap("train_txt", tr_txts, "val_txt", va_txts)
    report_overlap("train_txt", tr_txts, "test_txt", te_txts)
    report_overlap("val_txt",   va_txts, "test_txt", te_txts)

    print("\n[LeakCheck] Done.\n")

def cleanSST(string):
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)

def train_val_test(window_size):
    config = Config()

    train_dataset = pickle.load(open('./pickles/new_train_dataset.pkl', 'rb'))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)


    model = Multi_Model(
        bert_path=config.bert_path,
        window_size=window_size
    )

    model.to(config.device)

    criterion = nn.CrossEntropyLoss()

    print("Start Stage 1 sanity check...")



    print("Stage 1 forward + loss DONE.")

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        input_ids, token_type_ids, attention_mask, images, event, labels = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(input_ids, attention_mask, images)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def evaluate(model, dataloader, device):
    model.eval()
    preds, labels_all = [], []
    sem_values, syn_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, token_type_ids, attention_mask, images, event, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask, images)

            if isinstance(outputs, tuple):

                logits = outputs[0]
                sem_gate = outputs[1] if len(outputs) > 1 else None
                syn_gate = outputs[2] if len(outputs) > 2 else None

                if sem_gate is not None:

                    sem_values.append(float(sem_gate.detach().cpu()))
                if syn_gate is not None:
                    syn_values.append(float(syn_gate.detach().cpu()))
            else:
                logits = outputs

            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())

    acc = accuracy_score(labels_all, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds, average="macro")

    if len(sem_values) > 0:
        print(f"[Analysis] Semantic gate avg: {sum(sem_values)/len(sem_values):.6f}, max: {max(sem_values):.6f}")
    if len(syn_values) > 0:
        print(f"[Analysis] Syntax gate avg: {sum(syn_values)/len(syn_values):.6f}, max: {max(syn_values):.6f}")

    return acc, precision, recall, f1


def visualize_tsne(model, dataloader, device, save_name="tsne.pdf"):
    print(f"[Visual] Generating t-SNE plot: {save_name} ...")
    model.eval()
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, token_type_ids, attention_mask, images, event, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            images = images.to(device)


            logits, features = model(input_ids, attention_mask, images, return_feature=True)

            all_feats.append(features.cpu().numpy())
            all_labels.append(labels.numpy())


    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)

    tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)


    plt.figure(figsize=(10, 8))

    colors = ['red' if label == 0 else 'blue' for label in y]


    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.6, s=15)


    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Fake News', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Real News', markerfacecolor='blue', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title("t-SNE Visualization of Multimodal Features")
    plt.xticks([])
    plt.yticks([])


    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visual] Saved to {save_name}")

def run_experiment(window_size, config_override=None, verbose_checks=False):
    """
    Paper-style protocol:
      - train/val/test
      - val for early stopping
      - final report on test
      - return a dict for ablation summary
    """
    config = config_override if config_override is not None else Config()

    # -------------------------
    # 1) Build datasets/loaders
    # -------------------------
    train_dataset = WeiboJsonDataset(
        json_path=os.path.join(DATA_ROOT, "train_datas.json"),
        bert_path=config.bert_path,
        max_len=128
    )
    val_dataset = WeiboJsonDataset(
        json_path=os.path.join(DATA_ROOT, "validate_datas.json"),
        bert_path=config.bert_path,
        max_len=128
    )

    test_json = os.path.join(DATA_ROOT, "test_datas.json")


    if os.path.exists(test_json):
        test_dataset = WeiboJsonDataset(
            json_path=test_json,
            bert_path=config.bert_path,
            max_len=128
        )
    else:
        # fallback (not recommended): use val as test
        print("[WARN] test_datas.json not found. Falling back to validate_datas.json as test (NOT paper-aligned).")
        test_dataset = val_dataset


    print("\n" + "="*70)
    print("[DataCheck] JSON paths")
    print("="*70)
    print("train_json:", os.path.join(DATA_ROOT, "train_datas.json"))
    print("val_json  :", os.path.join(DATA_ROOT, "validate_datas.json"))
    print("test_json :", os.path.join(DATA_ROOT, "test_datas.json"))


    from collections import Counter
    def quick_stats(ds, name, N=50):
        N = min(N, len(ds))
        ys = []
        lens = []
        zero_imgs = 0
        for i in range(N):
            input_ids, _, attn, img, _, y = ds[i]
            ys.append(int(y))
            lens.append(int(attn.sum().item()))
            if float(img.max()) == 0.0:
                zero_imgs += 1
        print(f"\n[DataCheck] {name} | size={len(ds)} | sampleN={N}")
        print("label_dist(sample):", Counter(ys))
        print("attn_len(sample): min/mean/max =",
              min(lens), sum(lens)/len(lens), max(lens))
        print("zero_img(sample):", zero_imgs, "/", N)

    quick_stats(train_dataset, "train")
    quick_stats(val_dataset, "val")
    quick_stats(test_dataset, "test")


    print("\n[DataCheck] missing_img counter (after sampling)")
    print("train missing:", getattr(train_dataset, "_missing_img", None))
    print("val   missing:", getattr(val_dataset, "_missing_img", None))
    print("test  missing:", getattr(test_dataset, "_missing_img", None))


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # -------------------------
    # 2) Optional sanity checks
    # -------------------------
    if verbose_checks:
        batch = next(iter(train_loader))
        batch_text0, _, _, batch_image, _, batch_label = batch
        print("[CHECK] input_ids:", batch_text0.shape)
        print("[CHECK] images:", batch_image.shape, batch_image.min().item(), batch_image.max().item())
        print("[CHECK] labels:", batch_label[:10])

        batch = next(iter(train_loader))
        batch_text0, _, batch_attn, batch_image, _, batch_label = batch
        print("[CHECK] input_ids:", batch_text0.shape)
        print("[CHECK] attn:", batch_attn.shape, batch_attn.sum(dim=1)[:5])
        print("[CHECK] images:", batch_image.shape, batch_image.min().item(), batch_image.max().item())
        print("[CHECK] labels:", batch_label[:10])

    # -------------------------
    # 3) Build model
    # -------------------------
    model = Multi_Model(
        bert_path=config.bert_path,
        window_size=window_size,
        use_window_graph=config.use_window_graph,
        use_semantic=config.use_semantic,
        use_syntax=config.use_syntax
    ).to(config.device)

    # one forward check (optional)
    if verbose_checks:
        model.eval()
        with torch.no_grad():
            batch = next(iter(train_loader))
            input_ids, _, attention_mask, images, _, labels = batch
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            images = images.to(config.device)
            out = model(input_ids, attention_mask, images)
            logits = out[0] if isinstance(out, tuple) else out
            print("[CHECK] logits:", logits.shape, "min/max", logits.min().item(), logits.max().item())

    # -------------------------
    # 4) Optimizer (paper: Adam, fixed lr)
    # -------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=getattr(config, "l2", 0.0))
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # 5) Train with early stopping on VAL (by F1)
    # -------------------------
    best_val_f1 = -1.0
    best_epoch = -1
    patience = getattr(config, "patience", 5)
    bad_epochs = 0
    best_state = None

    print(f"\n=== Training (paper-style) | window_size={window_size} | "
          f"window_graph={config.use_window_graph} semantic={config.use_semantic} syntax={config.use_syntax} ===")
    print(f"[Config] bs={config.batch_size} lr={config.lr} epochs={config.epochs} patience={patience} device={config.device}")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.device)

        val_acc, val_p, val_r, val_f1 = evaluate(model, val_loader, config.device)

        print(
            f"Epoch [{epoch + 1}/{config.epochs}] "
            f"Loss: {train_loss:.4f} | "
            f"VAL Acc: {val_acc:.4f} | P: {val_p:.4f} | R: {val_r:.4f} | F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            bad_epochs = 0
            # store best weights on CPU (safe across devices)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[EarlyStop] No VAL F1 improvement for {patience} epochs. Stop at epoch {epoch+1}.")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict({k: v.to(config.device) for k, v in best_state.items()})
    print(f"[BEST VAL] Epoch={best_epoch} | F1={best_val_f1:.4f}")

    # -------------------------
    # 6) Final TEST report (paper tables use TEST)
    # -------------------------
    test_acc, test_p, test_r, test_f1 = evaluate(model, test_loader, config.device)
    print(f"[TEST] Acc={test_acc:.4f} | P={test_p:.4f} | R={test_r:.4f} | F1={test_f1:.4f}")


    if config.use_semantic and config.use_syntax:
        visualize_tsne(model, test_loader, config.device, save_name="tsne_full_model.pdf")

    # return for ablation summary
    return {
        "val_best_epoch": best_epoch,
        "val_best_f1": best_val_f1,
        "test_acc": test_acc,
        "test_p": test_p,
        "test_r": test_r,
        "test_f1": test_f1,
    }





train_acc_vector = []
vali_acc_vector = []

if __name__ == "__main__":

    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    print("device:", Config().device,
          "cuda:", torch.cuda.is_available(),
          "mps:", torch.backends.mps.is_available())


    leakage_check(
        train_json=os.path.join(DATA_ROOT, "train_datas.json"),
        val_json=os.path.join(DATA_ROOT, "validate_datas.json"),
        test_json=os.path.join(DATA_ROOT, "test_datas.json"),
    )

    ablations = [
        #("T-only",  False, False),
        ("S-only",  True,  False),
        ("Y-only",  False, True),
        ("Full",    True,  True),
    ]

    WINDOW_SIZE = 0
    all_results = []

    for i, (name, use_sem, use_syn) in enumerate(ablations):
        print("\n" + "=" * 70)
        print(f"[Ablation] {name} | window_graph=False | semantic={use_sem} | syntax={use_syn}")
        print("=" * 70)

        cfg = Config()
        cfg.use_window_graph = False
        cfg.use_semantic = use_sem
        cfg.use_syntax = use_syn
        res = run_experiment(
            window_size=WINDOW_SIZE,
            config_override=cfg,
            verbose_checks=(i == 0)  # 只在第一个 ablation 打印一次详细检查
        )

        all_results.append((name, res))

    print("\n" + "=" * 70)
    print("Ablation Summary (TEST)")
    print("=" * 70)
    print(f"{'Model':<10} | {'Acc':>7} | {'P':>7} | {'R':>7} | {'F1':>7} | {'BestVAL_Epoch':>12} | {'BestVAL_F1':>10}")
    print("-" * 86)
    for name, r in all_results:
        print(
            f"{name:<10} | {r['test_acc']:>7.4f} | {r['test_p']:>7.4f} | {r['test_r']:>7.4f} | {r['test_f1']:>7.4f} | "
            f"{r['val_best_epoch']:>12} | {r['val_best_f1']:>10.4f}"
        )
