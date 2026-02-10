import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter

from mymodel import FullGraphAttnMP, SemanticGAUFusion, SAFusionClassifier


# =========================
# Config
# =========================
seed = 3407
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

TWITTER_ROOT = "/Users/yanyuhan/Downloads/Twitter_Rumor_Detection"

class Config:
    def __init__(self):
        self.batch_size = 4
        self.epochs = 3
        self.lr = 2e-5
        self.l2 = 1e-5
        self.max_len = 128


        self.bert_path = "bert-base-uncased"

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")


        self.use_window_graph = False
        self.use_semantic = True
        self.use_syntax = True
        self.patience = 5
        self.val_ratio = 0.15


# =========================
# Utils: npz key auto-detect
# =========================
def _pick_first_key(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    return None

def _to_str_list(x):
    """
    x might be:
      - numpy array of dtype <U / object (bytes)
      - list
    """
    if isinstance(x, list):
        return [str(i) for i in x]
    arr = np.array(x)
    if arr.dtype == object:
        out = []
        for v in arr:
            if isinstance(v, (bytes, bytearray)):
                out.append(v.decode("utf-8", errors="ignore"))
            else:
                out.append(str(v))
        return out
    if arr.dtype.kind in ["S"]:  # bytes
        return [v.decode("utf-8", errors="ignore") for v in arr.tolist()]
    return [str(v) for v in arr.tolist()]

def _ensure_long_labels(y):
    y = np.array(y).reshape(-1)

    if y.ndim == 1:
        return y.astype(np.int64)
    # one-hot -> argmax
    return np.argmax(y, axis=-1).astype(np.int64)

def _ensure_image_tensor(img_arr):
    """
    Return torch.FloatTensor [3,224,224] in 0~1.
    Handles common cases:
      - [H,W,3] uint8
      - [3,H,W] uint8/float
      - [H,W] grayscale -> repeat
      - [N,H,W,3] handled outside per-sample
    """
    x = np.array(img_arr)

    if x.ndim == 2:  # H,W -> grayscale
        x = np.stack([x, x, x], axis=-1)  # H,W,3

    if x.ndim == 3 and x.shape[-1] == 3:      # H,W,3
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
    elif x.ndim == 3 and x.shape[0] == 3:     # 3,H,W
        x = torch.from_numpy(x).contiguous()
    else:
        raise ValueError(f"Unrecognized image shape: {x.shape}")

    x = x.float()

    if x.max().item() > 1.5:
        x = x / 255.0


    if x.shape[-2:] != (224, 224):
        x = x.unsqueeze(0)  # [1,3,H,W]
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.squeeze(0)

    return x.clamp(0.0, 1.0)


# =========================
# Dataset
# =========================
class TwitterFeatureDataset(Dataset):

    def __init__(self, text_npz_path, image_npz_path, debug_print=True):
        text_npz = np.load(text_npz_path, allow_pickle=True)
        img_npz  = np.load(image_npz_path, allow_pickle=True)

        if "data" not in text_npz.files or "label" not in text_npz.files:
            raise ValueError(f"text_npz keys={text_npz.files} (expect 'data','label')")
        if "data" not in img_npz.files:
            raise ValueError(f"image_npz keys={img_npz.files} (expect 'data')")

        self.text_data = text_npz["data"]   # expected [N,L,Dt] or object list of [L,Dt]
        self.labels    = text_npz["label"].reshape(-1).astype(np.int64)

        self.img_data  = img_npz["data"]    # expected [N,512,1,1]

        # clamp length if mismatch
        n = min(len(self.labels), len(self.text_data), len(self.img_data))
        self.text_data = self.text_data[:n]
        self.labels = self.labels[:n]
        self.img_data = self.img_data[:n]

        self._printed = False

        if debug_print:
            print("\n" + "="*70)
            print("[TwitterFeatureDataset] Loaded")
            print("="*70)
            print("text_npz :", text_npz_path)
            print("image_npz:", image_npz_path)
            print("text_data.shape:", np.array(self.text_data).shape, "dtype:", np.array(self.text_data).dtype)
            print("img_data.shape :", np.array(self.img_data).shape,  "dtype:", np.array(self.img_data).dtype)
            print("labels.shape   :", self.labels.shape)

            # print one sample
            x0 = self.text_data[0]
            print("[DEBUG] one text sample shape:", np.array(x0).shape, "dtype:", np.array(x0).dtype)
            print("[DEBUG] one image sample shape:", np.array(self.img_data[0]).shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        txt = np.array(self.text_data[idx], dtype=np.float32)   # [L,Dt]
        if txt.ndim != 2:
            raise ValueError(f"text feature should be 2D [L,Dt], got {txt.shape}")

        # mask: a row is valid if not all zeros
        row_abs = np.abs(txt).sum(axis=1)  # [L]
        mask = (row_abs > 0).astype(np.int64)  # [L]

        img = np.array(self.img_data[idx], dtype=np.float32)  # [512,1,1] or [512]
        if img.ndim == 3 and img.shape[1:] == (1,1):
            img = img.reshape(-1)  # [512]
        elif img.ndim == 1:
            pass
        else:
            raise ValueError(f"image feature unexpected shape: {img.shape}")

        text_feat = torch.from_numpy(txt)              # [L,Dt]
        text_mask = torch.from_numpy(mask)             # [L]
        img_feat  = torch.from_numpy(img)              # [512]
        label     = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        if (not self._printed) and idx < 3:
            self._printed = True
            print("\n[SampleCheck-TwitterFeature]")
            print("idx:", idx, "label:", int(label))
            print("text_feat:", text_feat.shape, "mask_valid:", int(text_mask.sum().item()))
            print("img_feat :", img_feat.shape, "min/max:", float(img_feat.min()), float(img_feat.max()))

        return text_feat, text_mask, img_feat, label


class TwitterFeatureModel(nn.Module):

    def __init__(self, text_dim, use_semantic=True, use_syntax=True, num_classes=2):
        super().__init__()
        self.use_semantic = use_semantic
        self.use_syntax = use_syntax

        # project to shared dim=128
        self.text_proj = nn.Linear(text_dim, 128)
        self.img_proj  = nn.Linear(512, 128)

        # paper-aligned blocks (reuse your mymodel modules)
        self.t_gnn1 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.t_gnn2 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.t_gnn3 = FullGraphAttnMP(dim=128, dropout=0.1)

        self.v_gnn1 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.v_gnn2 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.v_gnn3 = FullGraphAttnMP(dim=128, dropout=0.1)

        self.mfm1 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.mfm2 = FullGraphAttnMP(dim=128, dropout=0.1)

        self.t_map = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.v_map = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))

        self.sem_gau = SemanticGAUFusion(dim=128)
        self.fusion_classifier = SAFusionClassifier(dim=128, num_classes=num_classes, dropout=0.1)

    def forward(self, text_feat, text_mask, img_feat):
        """
        returns:
          train: logits
          eval : logits, sem_alpha_mean, None
        """
        B, L, Dt = text_feat.shape
        sem_alpha_mean = None

        # 1) project text token features
        t = self.text_proj(text_feat)  # [B,L,128]
        t_mask = text_mask             # [B,L]

        # pooled text baseline for semantic
        mm = t_mask.unsqueeze(-1).float()
        text_pool = (t * mm).sum(dim=1) / (mm.sum(dim=1).clamp_min(1.0))  # [B,128]

        # 2) image global feature
        img_repr = self.img_proj(img_feat)  # [B,128]

        # defaults
        g_vt_vec = torch.zeros(B, 128, device=t.device)
        s_vt_vec = torch.zeros(B, 128, device=t.device)

        # 3) syntax branch (graph)
        if self.use_syntax:
            t2 = self.t_gnn1(t, t_mask)
            t2 = self.t_gnn2(t2, t_mask)
            t2 = self.t_gnn3(t2, t_mask)

            # visual nodes: only one node v0 (since no patch nodes in this dataset)
            v = img_repr.unsqueeze(1)  # [B,1,128]
            v_mask = None
            v = self.v_gnn1(v, v_mask)
            v = self.v_gnn2(v, v_mask)
            v = self.v_gnn3(v, v_mask)

            t2 = self.t_map(t2)
            v  = self.v_map(v)

            nodes = torch.cat([t2, v], dim=1)  # [B,L+1,128]

            # multimodal mask: [B,L+1]
            v_valid = torch.ones(B, 1, device=t_mask.device, dtype=t_mask.dtype)
            mm_mask = torch.cat([t_mask, v_valid], dim=1)

            nodes = self.mfm1(nodes, mm_mask)
            nodes = self.mfm2(nodes, mm_mask)

            mmf = mm_mask.unsqueeze(-1).float()
            g_vt_vec = (nodes * mmf).sum(dim=1) / (mmf.sum(dim=1).clamp_min(1.0))  # [B,128]

        # 4) semantic GAU
        if self.use_semantic:
            s_vt_vec, sem_alpha_mean = self.sem_gau(text_pool, img_repr)

        # 5) classifier
        logits = self.fusion_classifier(g_vt_vec, s_vt_vec)

        if self.training:
            return logits
        else:
            return logits, sem_alpha_mean, None





def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        text_feat, text_mask, img_feat, labels = batch
        text_feat = text_feat.to(device)
        text_mask = text_mask.to(device)
        img_feat  = img_feat.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(text_feat, text_mask, img_feat)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def evaluate(model, dataloader, device):
    model.eval()
    preds, labels_all = [], []
    sem_values = []

    with torch.no_grad():
        for batch in dataloader:
            text_feat, text_mask, img_feat, labels = batch
            text_feat = text_feat.to(device)
            text_mask = text_mask.to(device)
            img_feat  = img_feat.to(device)
            labels    = labels.to(device)

            outputs = model(text_feat, text_mask, img_feat)

            if isinstance(outputs, tuple):
                logits = outputs[0]
                sem_gate = outputs[1]
                if sem_gate is not None:
                    sem_values.append(float(sem_gate.detach().cpu()))
            else:
                logits = outputs

            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())

    acc = accuracy_score(labels_all, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds, average="macro")

    if len(sem_values) > 0:
        print(f"[Analysis] Semantic gate avg: {sum(sem_values)/len(sem_values):.6f}, max: {max(sem_values):.6f}")

    return acc, precision, recall, f1



def run_experiment_twitter(config: Config, verbose_checks=True):
    train_text = os.path.join(TWITTER_ROOT, "train_text_with_label.npz")
    train_img  = os.path.join(TWITTER_ROOT, "train_image_with_label.npz")
    test_text  = os.path.join(TWITTER_ROOT, "test_text_with_label.npz")
    test_img   = os.path.join(TWITTER_ROOT, "test_image_with_label.npz")

    full_train = TwitterFeatureDataset(train_text, train_img, debug_print=True)
    test_ds    = TwitterFeatureDataset(test_text,  test_img,  debug_print=True)

    # split train -> train/val
    n = len(full_train)
    idxs = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idxs)

    val_n = int(n * config.val_ratio)
    val_idx = idxs[:val_n].tolist()
    tr_idx  = idxs[val_n:].tolist()

    train_ds = Subset(full_train, tr_idx)
    val_ds   = Subset(full_train, val_idx)

    def quick_label_stats(ds, name, N=200):
        N = min(N, len(ds))
        ys = []
        zero_imgs = 0
        for i in range(N):
            text_feat, text_mask, img_feat, y = ds[i]
            ys.append(int(y))
            if float(img_feat.abs().max()) == 0.0:
                zero_imgs += 1
        print(f"\n[DataCheck-Twitter] {name} | size={len(ds)} | sampleN={N}")
        print("label_dist(sample):", Counter(ys))
        print("zero_img_feat(sample):", zero_imgs, "/", N)

    quick_label_stats(train_ds, "train")
    quick_label_stats(val_ds, "val")
    quick_label_stats(test_ds, "test")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False, num_workers=0)

    # infer text_dim
    x0, m0, v0, y0 = full_train[0]
    text_dim = x0.shape[-1]

    model = TwitterFeatureModel(
        text_dim=text_dim,
        use_semantic=config.use_semantic,
        use_syntax=config.use_syntax,
        num_classes=2
    ).to(config.device)

    if verbose_checks:
        model.eval()
        with torch.no_grad():
            batch = next(iter(train_loader))
            text_feat, text_mask, img_feat, labels = batch
            text_feat = text_feat.to(config.device)
            text_mask = text_mask.to(config.device)
            img_feat  = img_feat.to(config.device)
            out = model(text_feat, text_mask, img_feat)
            logits = out[0] if isinstance(out, tuple) else out
            print("[CHECK] logits:", logits.shape, "min/max", logits.min().item(), logits.max().item())

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    best_state = None

    print(f"\n=== Training (Twitter-Feature) | semantic={config.use_semantic} syntax={config.use_syntax} ===")
    print(f"[Config] bs={config.batch_size} lr={config.lr} epochs={config.epochs} patience={config.patience} device={config.device}")

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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print(f"[EarlyStop] No VAL F1 improvement for {config.patience} epochs. Stop at epoch {epoch+1}.")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(config.device) for k, v in best_state.items()})

    print(f"[BEST VAL] Epoch={best_epoch} | F1={best_val_f1:.4f}")

    test_acc, test_p, test_r, test_f1 = evaluate(model, test_loader, config.device)
    print(f"[TEST] Acc={test_acc:.4f} | P={test_p:.4f} | R={test_r:.4f} | F1={test_f1:.4f}")

    return {
        "val_best_epoch": best_epoch,
        "val_best_f1": best_val_f1,
        "test_acc": test_acc,
        "test_p": test_p,
        "test_r": test_r,
        "test_f1": test_f1,
    }



if __name__ == "__main__":
    cfg = Config()
    print("device:", cfg.device, "cuda:", torch.cuda.is_available(), "mps:", torch.backends.mps.is_available())

    ablations = [
        ("T-only",  False, False),
        ("S-only",  True,  False),
        ("Y-only",  False, True),
        ("Full",    True,  True),
    ]

    all_results = []
    for i, (name, use_sem, use_syn) in enumerate(ablations):
        print("\n" + "=" * 70)
        print(f"[Ablation] {name} | semantic={use_sem} | syntax={use_syn}")
        print("=" * 70)

        cfg = Config()
        cfg.use_window_graph = False
        cfg.use_semantic = use_sem
        cfg.use_syntax = use_syn

        res = run_experiment_twitter(cfg, verbose_checks=(i == 0))

        all_results.append((name, res))

    print("\n" + "=" * 70)
    print("Ablation Summary (Twitter TEST)")
    print("=" * 70)
    print(f"{'Model':<10} | {'Acc':>7} | {'P':>7} | {'R':>7} | {'F1':>7} | {'BestVAL_Epoch':>12} | {'BestVAL_F1':>10}")
    print("-" * 86)
    for name, r in all_results:
        print(
            f"{name:<10} | {r['test_acc']:>7.4f} | {r['test_p']:>7.4f} | {r['test_r']:>7.4f} | {r['test_f1']:>7.4f} | "
            f"{r['val_best_epoch']:>12} | {r['val_best_f1']:>10.4f}"
        )
