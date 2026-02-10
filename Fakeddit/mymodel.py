import torch
import torch.nn as nn
from transformers import BertModel
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
    def __init__(self, hidden_dim):
        super().__init__()
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)

    def forward(self, node_feats, adj):
        x = self.gcn1(node_feats, adj)
        x = self.gcn2(x, adj)
        return x


class CrossModalAttnMP(nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.msg_t_from_o = nn.Linear(dim, dim)
        self.msg_o_from_t = nn.Linear(dim, dim)
        self.upd_t = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU())
        self.upd_o = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU())
        self.ln_t = nn.LayerNorm(dim)
        self.ln_o = nn.LayerNorm(dim)

    def forward(self, token_nodes, obj_nodes, attn_to_obj):
        msg_o = self.msg_t_from_o(obj_nodes)
        agg_to_t = torch.bmm(attn_to_obj, msg_o)
        attn_to_tok = attn_to_obj.transpose(1, 2).contiguous()
        msg_t = self.msg_o_from_t(token_nodes)
        agg_to_o = torch.bmm(attn_to_tok, msg_t)
        new_t = self.upd_t(torch.cat([token_nodes, agg_to_t], dim=-1))
        new_o = self.upd_o(torch.cat([obj_nodes, agg_to_o], dim=-1))
        new_t = self.ln_t(token_nodes + self.dropout(new_t))
        new_o = self.ln_o(obj_nodes + self.dropout(new_o))
        return new_t, new_o


class TextWindowAttnMP(nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.upd = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU())
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, mask=None):
        B, L, D = x.shape
        q = self.q(x);
        k = self.k(x);
        v = self.v(x)
        logits = torch.matmul(q, k.transpose(1, 2)) / (D ** 0.5)
        logits = logits.masked_fill(adj == 0, -1e9)
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            logits = logits.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        agg = torch.bmm(attn, v)
        out = self.upd(torch.cat([x, agg], dim=-1))
        out = self.ln(x + self.dropout(out))
        return out


class FullGraphAttnMP(nn.Module):
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
        q = self.q(x);
        k = self.k(x);
        v = self.v(x)
        logits = torch.matmul(q, k.transpose(1, 2)) / (D ** 0.5)
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            logits = logits.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        agg = torch.bmm(attn, v)
        out = self.upd(torch.cat([x, agg], dim=-1))
        out = self.ln(x + self.dropout(out))
        return out


class GAU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q_vec, kv_vec):
        Q = self.Wq(q_vec);
        K = self.Wk(kv_vec);
        V = self.Wv(kv_vec)
        attn_score = (Q * K).sum(dim=-1, keepdim=True)
        attn_weight = torch.sigmoid(attn_score)
        out = attn_weight * V
        out = self.Wo(self.dropout(out))
        return out, attn_weight.mean()


class SemanticGAUFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gau = GAU(dim)

    def forward(self, S_t, S_v):
        a1, g1 = self.gau(S_v, S_t)
        a2, g2 = self.gau(S_t, S_v)
        S_h = a1 + a2
        b1, g3 = self.gau(S_h, S_t)
        b2, g4 = self.gau(S_t, S_h)
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
            nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(dim)
        self.cls = nn.Linear(dim, num_classes)

    def forward(self, g_vec, s_vec):
        g = self.g_proj(g_vec)
        s = self.s_proj(s_vec)
        x = torch.stack([g, s], dim=1)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_logits = torch.matmul(q, k.transpose(1, 2)) / (self.dim ** 0.5)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = self.ln(out + self.ffn(out))
        fused = out.mean(dim=1)
        logits = self.cls(fused)
        return logits, fused



class Multi_Model(nn.Module):
    def __init__(self, bert_path, window_size, num_classes=2,
                 use_window_graph=True, use_semantic=True, use_syntax=True):
        super().__init__()
        self.use_window_graph = use_window_graph
        self.use_semantic = use_semantic
        self.use_syntax = use_syntax
        self.window_size = window_size

        # BERT
        self.bert = BertModel.from_pretrained(bert_path)

        # Projections
        self.graph_proj = nn.Linear(self.bert.config.hidden_size, 256)
        self.text_proj = nn.Linear(self.bert.config.hidden_size, 128)

        # Swin Transformer
        from torchvision.models import swin_t, Swin_T_Weights
        from torchvision.models.feature_extraction import create_feature_extractor
        swin_weights = Swin_T_Weights.DEFAULT
        swin = swin_t(weights=swin_weights)
        self.swin_extractor = create_feature_extractor(swin, return_nodes={"features.7": "feat"})
        for p in self.swin_extractor.parameters():
            p.requires_grad = False
        self.swin_out_dim = 768
        self.obj_proj = nn.Linear(self.swin_out_dim, 128)
        self.obj_dropout = nn.Dropout(0.3)
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_proj = nn.Linear(self.swin_out_dim, 128)
        self.img_dropout = nn.Dropout(0.3)
        self.swin_preprocess = swin_weights.transforms()

        # Frequency VGG
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        features = vgg.features
        first = features[0]
        new_first = nn.Conv2d(6, first.out_channels, first.kernel_size, first.stride, first.padding,
                              bias=(first.bias is not None))
        with torch.no_grad():
            new_first.weight[:, :3] = first.weight
            new_first.weight[:, 3:] = first.weight
            if first.bias is not None: new_first.bias.copy_(first.bias)
        features[0] = new_first
        self.freq_vgg = features
        for p in self.freq_vgg.parameters():
            p.requires_grad = False
        self.freq_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.freq_proj = nn.Linear(512, 128)
        self.freq_dropout = nn.Dropout(0.3)

        # Gates & Fusion
        self.syntax_gate = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        with torch.no_grad():
            nn.init.constant_(self.syntax_gate[0].bias, -1.0)
        self.graph_scale = nn.Parameter(torch.tensor(0.1))
        self.sem_gau = SemanticGAUFusion(dim=128)
        self.sem_freq_fuse = nn.Sequential(nn.Linear(128 * 2, 128), nn.ReLU(), nn.Dropout(0.1))

        # GNNs
        self.t_gnn1 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.t_gnn2 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.t_gnn3 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.v_gnn1 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.v_gnn2 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.v_gnn3 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.mfm1 = FullGraphAttnMP(dim=128, dropout=0.1)
        self.mfm2 = FullGraphAttnMP(dim=128, dropout=0.1)

        # Window Graph
        self.win_attn1 = TextWindowAttnMP(dim=256, dropout=0.1)
        self.win_attn2 = TextWindowAttnMP(dim=128, dropout=0.1)
        self.win_proj_256_to_128 = nn.Linear(256, 128)
        self.graph_dropout = nn.Dropout(p=0.3)
        self.graph_gate = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

        self.t_map = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.v_map = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.fusion_classifier = SAFusionClassifier(dim=128, num_classes=num_classes, dropout=0.1)

    def forward(self, input_ids, attention_mask=None, images=None, return_feature=False):
        assert images is not None
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        tokens = outputs.last_hidden_state
        B, L, _ = tokens.size()

        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            text_pool = (tokens * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
        else:
            text_pool = tokens.mean(dim=1)
        text_repr = self.text_proj(text_pool)
        text_repr_base = text_repr

        # Window Graph (Optional)
        if self.use_window_graph and (self.window_size > 0):
            adj_txt = build_window_adj(B, L, self.window_size, tokens.device)
            if attention_mask is not None:
                mask2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                adj_txt = adj_txt * mask2d
            x = self.graph_proj(tokens)
            x = self.win_attn1(x, adj_txt, attention_mask)
            x = self.win_proj_256_to_128(x)
            x = self.graph_dropout(x)
            x = self.win_attn2(x, adj_txt, attention_mask)
            x = self.graph_dropout(x)
            if attention_mask is not None:
                m = attention_mask.unsqueeze(-1).float()
                graph_repr = (x * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
            else:
                graph_repr = x.mean(dim=1)
            gate = self.graph_gate(graph_repr)
            text_repr = text_repr + self.graph_scale * gate * graph_repr

        # Images
        g_vt_vec = torch.zeros(B, 128, device=tokens.device)
        s_vt_vec = torch.zeros(B, 128, device=tokens.device)
        freq_repr = torch.zeros(B, 128, device=tokens.device)
        sem_alpha_mean = None
        syntax_alpha_mean = None

        if images is not None:
            # Swin
            imgs_swin = images
            # Manual normalization using swin stats
            mean = torch.tensor(self.swin_preprocess.mean, device=imgs_swin.device).view(1, 3, 1, 1)
            std = torch.tensor(self.swin_preprocess.std, device=imgs_swin.device).view(1, 3, 1, 1)
            imgs_swin = (imgs_swin - mean) / std
            feat_map = self.swin_extractor(imgs_swin)["feat"]
            if feat_map.dim() == 4 and feat_map.shape[-1] == self.swin_out_dim:
                feat_map = feat_map.permute(0, 3, 1, 2).contiguous()

            spatial_global = self.img_pool(feat_map).squeeze(-1).squeeze(-1)
            img_repr = self.img_proj(spatial_global)
            img_repr = self.img_dropout(img_repr)

            C = feat_map.shape[1]
            obj_nodes = feat_map.view(B, C, -1).permute(0, 2, 1).contiguous()
            obj_nodes = self.obj_proj(obj_nodes)
            obj_nodes = self.obj_dropout(obj_nodes)

            # Freq
            fft = torch.fft.fft2(images, dim=(-2, -1))
            freq_in = torch.cat([fft.real, fft.imag], dim=1)
            freq_in = freq_in / (freq_in.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6))
            freq_map = self.freq_vgg(freq_in)
            freq_global = self.freq_pool(freq_map).squeeze(-1).squeeze(-1)
            freq_repr = self.freq_proj(freq_global)
            freq_repr = self.freq_dropout(freq_repr)

            # Syntax
            if self.use_syntax:
                token_nodes = self.text_proj(tokens)
                t = token_nodes;
                t_mask = attention_mask
                t = self.t_gnn1(t, t_mask);
                t = self.t_gnn2(t, t_mask);
                t = self.t_gnn3(t, t_mask)

                v0 = img_repr.unsqueeze(1)
                v = torch.cat([v0, obj_nodes], dim=1)
                v = self.v_gnn1(v);
                v = self.v_gnn2(v);
                v = self.v_gnn3(v)

                t = self.t_map(t);
                v = self.v_map(v)
                nodes = torch.cat([t, v], dim=1)
                if t_mask is not None:
                    v_valid = torch.ones(B, v.size(1), device=t_mask.device)
                    mm_mask = torch.cat([t_mask, v_valid], dim=1)
                else:
                    mm_mask = None
                nodes = self.mfm1(nodes, mm_mask);
                nodes = self.mfm2(nodes, mm_mask)

                if mm_mask is not None:
                    mm = mm_mask.unsqueeze(-1).float()
                    g_vt_vec = (nodes * mm).sum(dim=1) / (mm.sum(dim=1).clamp_min(1.0))
                else:
                    g_vt_vec = nodes.mean(dim=1)
                syntax_alpha = self.syntax_gate(g_vt_vec)
                syntax_alpha_mean = syntax_alpha.mean()

            # Semantic
            if self.use_semantic:
                s_vt_vec, sem_alpha_mean = self.sem_gau(text_repr_base, img_repr)
                s_vt_vec = self.sem_freq_fuse(torch.cat([s_vt_vec, freq_repr], dim=-1))

        pred, fused_feat = self.fusion_classifier(g_vt_vec, s_vt_vec)

        if return_feature:
            return pred, fused_feat
        if self.training:
            return pred
        else:
            return pred, sem_alpha_mean, syntax_alpha_mean