import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from gcn import GCNLayer




def build_window_adj(batch_size, seq_len, window_size, device):

    adj = torch.zeros(batch_size, seq_len, seq_len, device=device)

    for offset in range(-window_size, window_size + 1):
        if offset == 0:
            continue
        i = torch.arange(seq_len, device=device)
        j = i + offset
        valid = (j >= 0) & (j < seq_len)
        adj[:, i[valid], j[valid]] = 1

    # self-loop
    adj += torch.eye(seq_len, device=device).unsqueeze(0)

    return adj



class CrossModalGCN(nn.Module):
    """
    Cross-modal GCN over [Text tokens + Image objects]
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)

    def forward(self, node_feats, adj):
        """
        node_feats: [B, L+N, D]
        adj:        [B, L+N, L+N]
        """
        x = self.gcn1(node_feats, adj)
        x = self.gcn2(x, adj)
        return x

class CrossModalAttnMP(nn.Module):
    """
    GAT-like cross-modal message passing WITHOUT torch_geometric.
    Uses token->object attention (and reverse) as edge weights.
    """
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # messages
        self.msg_t_from_o = nn.Linear(dim, dim)
        self.msg_o_from_t = nn.Linear(dim, dim)

        # update
        self.upd_t = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU())
        self.upd_o = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU())

        # layernorm for stability
        self.ln_t = nn.LayerNorm(dim)
        self.ln_o = nn.LayerNorm(dim)

    def forward(self, token_nodes, obj_nodes, attn_to_obj):
        """
        token_nodes: [B, L, D]
        obj_nodes:   [B, N, D]
        attn_to_obj: [B, L, N]  (row-normalized, sparse allowed)
        """


        B, L, D = token_nodes.shape
        _, N, _ = obj_nodes.shape

        # token <- objects (weighted sum over objects)
        # msg_o: [B, N, D] -> [B, N, D]
        msg_o = self.msg_t_from_o(obj_nodes)                       # [B, N, D]
        agg_to_t = torch.bmm(attn_to_obj, msg_o)                   # [B, L, D]

        # object <- tokens (use reverse attention)
        # attn_to_tok: [B, N, L]
        attn_to_tok = attn_to_obj.transpose(1, 2).contiguous()     # [B, N, L]
        msg_t = self.msg_o_from_t(token_nodes)                     # [B, L, D]
        agg_to_o = torch.bmm(attn_to_tok, msg_t)                   # [B, N, D]

        # update with residual-style fusion
        new_t = self.upd_t(torch.cat([token_nodes, agg_to_t], dim=-1))
        new_o = self.upd_o(torch.cat([obj_nodes,   agg_to_o], dim=-1))

        new_t = self.ln_t(token_nodes + self.dropout(new_t))
        new_o = self.ln_o(obj_nodes   + self.dropout(new_o))

        return new_t, new_o

class TextWindowAttnMP(nn.Module):
    """
    GAT-like message passing on text tokens using a fixed window adjacency.
    No torch_geometric required.

    tokens: [B, L, D]
    adj:    [B, L, L]  (0/1, includes self-loops)
    mask:   [B, L]     (attention_mask, 1=valid, 0=pad)
    """
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.upd = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU())
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, mask=None):
        # x: [B,L,D], adj: [B,L,L]
        B, L, D = x.shape

        q = self.q(x)  # [B,L,D]
        k = self.k(x)  # [B,L,D]
        v = self.v(x)  # [B,L,D]

        # attention logits: [B,L,L]
        logits = torch.matmul(q, k.transpose(1, 2)) / (D ** 0.5)

        # 1) apply window adjacency: where adj==0, block attention
        logits = logits.masked_fill(adj == 0, -1e9)

        # 2) apply padding mask (don’t attend from/to PAD tokens)
        if mask is not None:
            # block queries from PAD: (optional, but helps stability)
            logits = logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            # block keys that are PAD
            logits = logits.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = torch.softmax(logits, dim=-1)  # [B,L,L]
        attn = self.dropout(attn)

        agg = torch.bmm(attn, v)  # [B,L,D]

        out = self.upd(torch.cat([x, agg], dim=-1))  # [B,L,D]
        out = self.ln(x + self.dropout(out))
        return out

class FullGraphAttnMP(nn.Module):
    """
    Single-head GAT-like message passing on a FULLY-CONNECTED graph.
    Equivalent to self-attention + residual update, but kept in "GNN wording".

    x:    [B, M, D]
    mask: [B, M]  (1=valid, 0=pad). If None -> all valid.
    """
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.upd = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU())
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, M, D = x.shape
        q = self.q(x)  # [B,M,D]
        k = self.k(x)  # [B,M,D]
        v = self.v(x)  # [B,M,D]

        logits = torch.matmul(q, k.transpose(1, 2)) / (D ** 0.5)  # [B,M,M]

        if mask is not None:
            # block queries from PAD
            logits = logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            # block keys that are PAD
            logits = logits.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = torch.softmax(logits, dim=-1)  # [B,M,M]
        attn = self.dropout(attn)

        agg = torch.bmm(attn, v)  # [B,M,D]

        out = self.upd(torch.cat([x, agg], dim=-1))  # [B,M,D]
        out = self.ln(x + self.dropout(out))
        return out




class GAU(nn.Module):
    """
    Vector-level Guided Attention Unit (paper-style).
    Q comes from q_vec, K/V come from kv_vec.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q_vec: torch.Tensor, kv_vec: torch.Tensor):
        """
        q_vec:  [B, D]  (query modality)
        kv_vec: [B, D]  (key/value modality)
        """
        # Project to Q, K, V
        Q = self.Wq(q_vec)           # [B, D]
        K = self.Wk(kv_vec)          # [B, D]
        V = self.Wv(kv_vec)          # [B, D]

        # -------- Guided Attention --------
        # vector-level attention score (dot-product)
        attn_score = (Q * K).sum(dim=-1, keepdim=True)  # [B, 1]

        # softmax over batch-wise scores (vector-level)
        attn_weight = torch.sigmoid(attn_score)  # [B, 1] 先用 sigmoid 保持每个样本独立
        out = attn_weight * V
        # [B, D]
        out = self.Wo(self.dropout(out))                # [B, D]

        # for logging / analysis
        alpha_mean = attn_weight.mean()

        return out, alpha_mean



class SemanticGAUFusion(nn.Module):
    """
    Two-round GAU interaction:
      S_h  = GAU(S_v, S_t) + GAU(S_t, S_v)
      S_vt = GAU(S_h, S_t) + GAU(S_t, S_h)
    Inputs: S_t, S_v: [B,D]
    Output: S_vt: [B,D], alpha_mean: scalar (avg of gate means)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gau = GAU(dim)

    def forward(self, S_t: torch.Tensor, S_v: torch.Tensor):
        a1, g1 = self.gau(S_v, S_t)   # GAU(S_v, S_t)
        a2, g2 = self.gau(S_t, S_v)   # GAU(S_t, S_v)
        S_h = a1 + a2

        b1, g3 = self.gau(S_h, S_t)   # GAU(S_h, S_t)
        b2, g4 = self.gau(S_t, S_h)   # GAU(S_t, S_h)
        S_vt = b1 + b2

        alpha_mean = (g1 + g2 + g3 + g4) / 4.0
        return S_vt, alpha_mean

class SAFusionClassifier(nn.Module):

    def __init__(self, dim=128, num_classes=2, dropout=0.1):
        super().__init__()
        self.dim = dim

        self.g_proj = nn.Linear(dim, dim)
        self.s_proj = nn.Linear(dim, dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attn_drop = nn.Dropout(dropout)


        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(dim)

        self.cls = nn.Linear(dim, num_classes)

    def forward(self, g_vec, s_vec):

        g = self.g_proj(g_vec)  # [B,dim]
        s = self.s_proj(s_vec)  # [B,dim]

        # 2 tokens
        x = torch.stack([g, s], dim=1)  # [B,2,dim]

        # QKV
        qkv = self.qkv(x)  # [B,2,3*dim]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B,2,dim]

        # Self-Attention (single-head)
        attn_logits = torch.matmul(q, k.transpose(1, 2)) / (self.dim ** 0.5)  # [B,2,2]
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [B,2,dim]

        # FFL + LN (residual)
        out = self.ln(out + self.ffn(out))  # [B,2,dim]


        fused = out.mean(dim=1)  # [B,dim]
        logits = self.cls(fused)  # [B,num_classes]
        return logits




class Multi_Model(nn.Module):
    def __init__(self, bert_path, window_size, num_classes=2,
                 use_window_graph=True, use_semantic=True, use_syntax=True):
        super().__init__()

        self.use_window_graph = use_window_graph
        self.use_semantic = use_semantic
        self.use_syntax = use_syntax

        self.sem_scale = nn.Parameter(torch.tensor(0.1))
        self.syn_scale = nn.Parameter(torch.tensor(0.1))

        self.cm_attn1 = CrossModalAttnMP(dim=128, dropout=0.1)
        self.cm_attn2 = CrossModalAttnMP(dim=128, dropout=0.1)

        self.window_size = window_size

        self.bert = BertModel.from_pretrained(bert_path)

        self.graph_proj = nn.Linear(
            self.bert.config.hidden_size, 256
        )

        self.syntax_gate = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )


        with torch.no_grad():
            nn.init.constant_(self.syntax_gate[0].bias, -1.0)


        self.graph_scale = nn.Parameter(torch.tensor(0.1))

        self.text_proj = nn.Linear(
            self.bert.config.hidden_size, 128
        )
        # ===== Image encoder backbone (VGG-19, pretrained) =====
        # ===== Image encoder (Swin Transformer, pretrained) =====
        from torchvision.models import swin_t, Swin_T_Weights
        from torchvision.models.feature_extraction import create_feature_extractor

        swin_weights = Swin_T_Weights.DEFAULT
        swin = swin_t(weights=swin_weights)


        self.swin_extractor = create_feature_extractor(
            swin,
            return_nodes={"features.7": "feat"}  # feat: [B, C, H, W]
        )

        for p in self.swin_extractor.parameters():
            p.requires_grad = False

        self.swin_out_dim = 768


        self.obj_proj = nn.Linear(self.swin_out_dim, 128)
        self.obj_dropout = nn.Dropout(0.3)

        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_proj = nn.Linear(self.swin_out_dim, 128)
        self.img_dropout = nn.Dropout(0.3)


        self.swin_preprocess = swin_weights.transforms()




        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)


        features = vgg.features
        first = features[0]  # Conv2d(3,64,3,1,1)

        new_first = nn.Conv2d(
            in_channels=6,
            out_channels=first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=(first.bias is not None)
        )

        with torch.no_grad():
            new_first.weight[:, :3] = first.weight
            new_first.weight[:, 3:] = first.weight
            if first.bias is not None:
                new_first.bias.copy_(first.bias)

        features[0] = new_first
        self.freq_vgg = features

        for p in self.freq_vgg.parameters():
            p.requires_grad = False

        self.freq_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.freq_proj = nn.Linear(512, 128)
        self.freq_dropout = nn.Dropout(0.5)

        self.freq_dropout = nn.Dropout(0.3)

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ===== Semantic gate (text-image consistency) =====

        self.sem_gate = nn.Sequential(
            nn.Linear(128 * 2, 1),
            nn.Sigmoid()
        )
        self.sem_gau = SemanticGAUFusion(dim=128)

        self.sem_freq_fuse = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )


        self.t_gnn1 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.t_gnn2 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.t_gnn3 = FullGraphAttnMP(dim=128, dropout=0.1)

        self.v_gnn1 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.v_gnn2 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.v_gnn3 = FullGraphAttnMP(dim=128, dropout=0.1)



        self.mfm1 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.mfm2 = FullGraphAttnMP(dim=128, dropout=0.1)


        # Text GCN
        # Text window attention (2 layers, GAT-like)
        self.win_attn1 = TextWindowAttnMP(dim=256, dropout=0.1)
        self.win_attn2 = TextWindowAttnMP(dim=128, dropout=0.1)
        self.win_proj_256_to_128 = nn.Linear(256, 128)

        self.graph_alpha = nn.Parameter(torch.tensor(0.0))

        self.fusion_classifier = SAFusionClassifier(dim=128, num_classes=num_classes, dropout=0.1)

        self.graph_dropout = nn.Dropout(p=0.3)
        self.graph_gate = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.t_map = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.v_map = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))

    def forward(self, input_ids, attention_mask=None, images=None):
        """
        Strict (paper-style) forward:
          1) Text encoder -> token features
          2) Intra-text window graph (optional)
          3) Image encoder -> object nodes (grid/patch nodes) + global image repr
          4) Cross-modal syntactic graph (token-object) + cross-modal GCN (optional)
          5) Cross-modal semantic gate (global text-image consistency) (optional)
          6) Fuse -> classifier
        """
        # ===== Allow ablations =====
        assert images is not None, "Images are required for this dataset/model."

        # Ch4 paper setting: no window graph
        assert (not self.use_window_graph), "Paper Ch4 does not use window graph; set use_window_graph=False."

        # =========================
        # 0) BERT encode text
        # =========================
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        tokens = outputs.last_hidden_state  # [B, L, 768]
        B, L, _ = tokens.size()


        if not hasattr(self, "_printed_syn_gate"):
            self._printed_syn_gate = False

        # =========================
        # 1) Text baseline (mask mean pool) -> [B, 128]
        # =========================
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            text_pool = (tokens * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))  # [B, 768]
        else:
            text_pool = tokens.mean(dim=1)

        text_repr = self.text_proj(text_pool)  # [B, 128]
        text_repr_base = text_repr  # ✅ 纯文本语义基准，供 semantic gate 用

        # ==========================================================
        # 2) Intra-text Window Graph (optional)  -> add graph_repr
        # ==========================================================
        if self.use_window_graph and (self.window_size > 0):
            # adj: [B, L, L]
            adj_txt = build_window_adj(
                batch_size=B, seq_len=L, window_size=self.window_size, device=tokens.device
            )
            if attention_mask is not None:
                mask2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # [B,L,L]
                adj_txt = adj_txt * mask2d

            x = self.graph_proj(tokens)  # [B, L, 256]

            # ---- window attention layer 1 (256-d) ----
            x = self.win_attn1(x, adj_txt, attention_mask)

            # ---- project to 128 ----
            x = self.win_proj_256_to_128(x)  # [B, L, 128]
            x = self.graph_dropout(x)

            # ---- window attention layer 2 (128-d) ----
            x = self.win_attn2(x, adj_txt, attention_mask)

            x = self.graph_dropout(x)

            if attention_mask is not None:
                m = attention_mask.unsqueeze(-1).float()
                graph_repr = (x * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))  # [B,128]
            else:
                graph_repr = x.mean(dim=1)  # [B,128]


            gate_logits_or_prob = self.graph_gate(graph_repr)  # [B,1]

            if gate_logits_or_prob.min().item() < 0.0 or gate_logits_or_prob.max().item() > 1.0:
                gate = torch.sigmoid(gate_logits_or_prob)
            else:
                gate = gate_logits_or_prob

            text_repr = text_repr + self.graph_scale * gate * graph_repr  # [B,128]

        # =========================
        # 3) Image features
        # =========================
        sem_alpha_mean = None  # for logging in eval
        syntax_alpha_mean = None

        # ablation fallback vectors (so classifier always has inputs)
        g_vt_vec = torch.zeros(B, 128, device=tokens.device)
        s_vt_vec = torch.zeros(B, 128, device=tokens.device)
        freq_repr = torch.zeros(B, 128, device=tokens.device)

        if images is not None:

            # ---- 3.1 Spatial branch (Swin) ----

            imgs_for_swin = images

            if imgs_for_swin.dim() == 4 and imgs_for_swin.shape[-1] == 3:
                imgs_for_swin = imgs_for_swin.permute(0, 3, 1, 2).contiguous()


            mean = torch.tensor(self.swin_preprocess.mean, device=imgs_for_swin.device).view(1, 3, 1, 1)
            std = torch.tensor(self.swin_preprocess.std, device=imgs_for_swin.device).view(1, 3, 1, 1)
            imgs_for_swin = (imgs_for_swin - mean) / std

            feat_map = self.swin_extractor(imgs_for_swin)["feat"]  # [B, C, H, W]



            if feat_map.dim() == 4 and feat_map.shape[-1] == self.swin_out_dim:
                feat_map = feat_map.permute(0, 3, 1, 2).contiguous()  # -> [B,C,H,W]

            B2, C, H, W = feat_map.shape
            assert B2 == B, "Batch mismatch between text and image"

            # global spatial vector S_v
            spatial_global = self.img_pool(feat_map).squeeze(-1).squeeze(-1)  # [B,C]
            img_repr = self.img_proj(spatial_global)  # [B,128]
            img_repr = self.img_dropout(img_repr)


            obj_nodes = feat_map.view(B, C, H * W).permute(0, 2, 1).contiguous()  # [B,N,C]
            obj_nodes = self.obj_proj(obj_nodes)  # [B,N,128]
            obj_nodes = self.obj_dropout(obj_nodes)
            N = obj_nodes.size(1)

            # ---- 3.2 Frequency branch (FFT magnitude) -> F_v ----

            fft = torch.fft.fft2(images, dim=(-2, -1))  # complex
            real = fft.real
            imag = fft.imag
            freq_in = torch.cat([real, imag], dim=1)  # [B,6,H,W]


            freq_in = freq_in / (freq_in.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6))

            freq_map = self.freq_vgg(freq_in)  # [B,512,h,w]
            freq_global = self.freq_pool(freq_map).squeeze(-1).squeeze(-1)  # [B,512]
            freq_repr = self.freq_proj(freq_global)  # [B,128]
            freq_repr = self.freq_dropout(freq_repr)

            # ==========================================================
            # 4) Cross-modal Syntactic Graph (paper Ch4)  -> g_vt_vec
            # ==========================================================
            if self.use_syntax:
                # Text nodes: token_nodes [B,L,128]
                token_nodes = self.text_proj(tokens)  # [B,L,128]
                t = token_nodes
                t_mask = attention_mask  # [B,L] (1 valid, 0 pad)

                # 4.4.1 T-GNN (3 layers)
                t = self.t_gnn1(t, t_mask)
                t = self.t_gnn2(t, t_mask)
                t = self.t_gnn3(t, t_mask)

                # Visual nodes: include global node v0 + patch nodes
                v0 = img_repr.unsqueeze(1)  # [B,1,128]
                v = torch.cat([v0, obj_nodes], dim=1)  # [B,1+N,128]
                v_mask = None  # all valid

                # 4.4.1 V-GNN (3 layers)
                v = self.v_gnn1(v, v_mask)
                v = self.v_gnn2(v, v_mask)
                v = self.v_gnn3(v, v_mask)

                # 4.4.2 MFM (2 layers) on full multimodal graph
                t = self.t_map(t)
                v = self.v_map(v)
                nodes = torch.cat([t, v], dim=1)  # [B, L+1+N, 128]

                if t_mask is not None:
                    v_valid = torch.ones(B, v.size(1), device=t_mask.device, dtype=t_mask.dtype)
                    mm_mask = torch.cat([t_mask, v_valid], dim=1)  # [B, L+1+N]
                else:
                    mm_mask = None

                nodes = self.mfm1(nodes, mm_mask)
                nodes = self.mfm2(nodes, mm_mask)

                # pool -> g_vt_vec
                if mm_mask is not None:
                    mm = mm_mask.unsqueeze(-1).float()
                    g_vt_vec = (nodes * mm).sum(dim=1) / (mm.sum(dim=1).clamp_min(1.0))
                else:
                    g_vt_vec = nodes.mean(dim=1)

                syntax_alpha = self.syntax_gate(g_vt_vec)  # [B,1]
                syntax_alpha_mean = syntax_alpha.mean()  # scalar

            # ==========================================================
            # 5) Cross-modal Semantic Gate (GAU) -> s_vt_vec
            # ==========================================================
            if self.use_semantic:
                # s_vt_vec: [B,128]
                s_vt_vec, sem_alpha_mean = self.sem_gau(text_repr_base, img_repr)
                s_vt_vec = self.sem_freq_fuse(torch.cat([s_vt_vec, freq_repr], dim=-1))

        # =========================
        # 6) Classifier
        # =========================
        # ===== Paper 4.5: SA fusion classifier =====
        # g_vt_vec from MFM pooled graph, s_vt_vec from GAU semantic interaction
        pred = self.fusion_classifier(g_vt_vec, s_vt_vec)

        if self.training:
            return pred
        else:
            return pred, sem_alpha_mean, syntax_alpha_mean


# =========================
# Supporting Modules
# =========================
class UnimodalDetection(nn.Module):
    def __init__(self, shared_dim=128, prime_dim=16):
        super().__init__()
        self.text_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),

            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )
        self.image_uni = nn.Se
