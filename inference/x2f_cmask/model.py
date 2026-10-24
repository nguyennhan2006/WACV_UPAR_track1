
import math, torch, torch.nn as nn, torch.nn.functional as F, torchvision
import torch.utils.checkpoint
import torchvision.transforms.functional as TF

IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP_NORM = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
MEAN, STD = IMAGENET_NORM            # giữ tương thích với bản cũ

SIGLIP_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def norm_for(arch):
    return CLIP_NORM if arch.startswith(("clip", "radio")) else SIGLIP_NORM if arch.startswith(("siglip", "pe_core")) else IMAGENET_NORM

def grayworld(t):
    # Cân bằng trắng gray-world trên tensor [0,1] (3,H,W): đưa trung bình 3 kênh về bằng nhau, giới hạn hệ số 0.6-1.6
    m = t.mean((1, 2)); g = (m.mean() / m.clamp(min=1e-3)).clamp(0.6, 1.6)
    return (t * g[:, None, None]).clamp(0, 1)

def crop_frac(img, crop):
    # 09/10: cắt theo tỷ lệ (x0, y0, x1, y1) của ảnh gốc, trước mọi phép resize (model chuyên vùng đầu). crop rỗng -> giữ nguyên ảnh
    if not crop: return img
    W, H = img.size; x0, y0, x1, y1 = crop
    l, t = int(round(x0 * W)), int(round(y0 * H))
    return img.crop((l, t, max(int(round(x1 * W)), l + 1), max(int(round(y1 * H)), t + 1)))

def letterbox(img, h, w, mean=IMAGENET_NORM[0]):
    # 11/10 (E31 lbox): giữ tỷ lệ khung — co ảnh vừa khít (w, h) rồi đệm 2 bên / trên dưới bằng màu trung bình (= 0 sau chuẩn hoá)
    s = min(w / img.width, h / img.height); nw, nh = max(1, round(img.width * s)), max(1, round(img.height * s))
    from PIL import Image as _I
    c = _I.new("RGB", (w, h), tuple(int(round(v * 255)) for v in mean)); c.paste(img.resize((nw, nh), resample=2), ((w - nw) // 2, (h - nh) // 2))
    return c

def preprocess(img, h, w, mean=IMAGENET_NORM[0], std=IMAGENET_NORM[1], wb=False, lbox=False):
    # PIL RGB -> tensor chuẩn hoá. Dùng chung cho val (notebook) và inference (run.py).
    img = letterbox(img, h, w, mean) if lbox else img.resize((w, h), resample=2)          # 2 = BILINEAR
    t = TF.to_tensor(img)
    if wb: t = grayworld(t)
    return TF.normalize(t, mean, std)

def _drop_path(x, p, training):
    if p == 0.0 or not training: return x
    keep = 1 - p
    return x * x.new_empty(x.shape[0], 1, 1).bernoulli_(keep) / keep

STRIPES = ((0.0, 0.30), (0.20, 0.60), (0.50, 1.00))      # tỷ lệ chiều cao: đầu, thân trên, thân dưới

def _stripe_means(tok, grid):
    # tok: (B, gh*gw, C) patch token theo thứ tự hàng -> trung bình từng dải ngang
    B, N, C = tok.shape; gh, gw = grid
    t = tok.reshape(B, gh, gw, C)
    return [t[:, int(round(a * gh)):max(int(round(a * gh)) + 1, int(round(b * gh)))].mean((1, 2)) for a, b in STRIPES]

def _resize_pos(grid_w, new_grid, antialias=True):
    # grid_w: (N, C) lưới vuông -> nội suy bicubic sang new_grid (h, w)
    s = int(math.sqrt(grid_w.shape[0])); C = grid_w.shape[1]
    g = grid_w.reshape(1, s, s, C).permute(0, 3, 1, 2).float()
    g = F.interpolate(g, size=new_grid, mode="bicubic", align_corners=False, antialias=antialias)
    return g.permute(0, 2, 3, 1).reshape(-1, C), s

# ------------------------------ DINOv2 ViT (key khớp checkpoint chính thức của Meta) ------------------------------
class _Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads
        self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))

class _Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.fc1 = nn.Linear(dim, hidden); self.act = nn.GELU(); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class _LayerScale(nn.Module):
    def __init__(self, dim): super().__init__(); self.gamma = nn.Parameter(torch.ones(dim) * 1e-5)
    def forward(self, x): return x * self.gamma

class _Block(nn.Module):
    def __init__(self, dim, heads, dp):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6); self.attn = _Attn(dim, heads); self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6); self.mlp = _Mlp(dim, dim * 4); self.ls2 = _LayerScale(dim)
        self.dp = dp
    def forward(self, x):
        x = x + _drop_path(self.ls1(self.attn(self.norm1(x))), self.dp, self.training)
        return x + _drop_path(self.ls2(self.mlp(self.norm2(x))), self.dp, self.training)

class _PatchEmbed(nn.Module):
    def __init__(self, dim, patch): super().__init__(); self.proj = nn.Conv2d(3, dim, patch, patch)
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)

class DinoViT(nn.Module):
    def __init__(self, dim, depth, heads, grid, n_reg=0, patch=14, drop_path=0.0, parts=False):
        super().__init__()
        self.grid, self.n_reg, self.parts = grid, n_reg, parts
        self.patch_embed = _PatchEmbed(dim, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid[0] * grid[1], dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, n_reg, dim)) if n_reg else None
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([_Block(dim, heads, d) for d in dps])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.out_dim = (2 + 3 * parts) * dim
    def features(self, x, taps=()):
        # (đặc trưng gộp, patch token cuối, patch token sau các khối trong taps)
        B = x.shape[0]
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.patch_embed(x)], 1) + self.pos_embed
        if self.n_reg:
            x = torch.cat([x[:, :1], self.register_tokens.expand(B, -1, -1), x[:, 1:]], 1)
        mids = []
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i in taps: mids.append(x[:, 1 + self.n_reg:])
        x = self.norm(x); pt = x[:, 1 + self.n_reg:]
        f = torch.cat([x[:, 0], pt.mean(1)] + (_stripe_means(pt, self.grid) if self.parts else []), 1)   # CLS + trung bình patch (+ 3 dải ngang)
        return f, pt, mids
    def forward(self, x, return_tokens=False):
        f, pt, _ = self.features(x)
        return (f, pt) if return_tokens else f

DINO_CFG = {"dinov2_vits14": (384, 12, 6), "dinov2_vitb14": (768, 12, 12), "dinov2_vitl14": (1024, 24, 16)}

def load_dinov2_pretrained(vit, name, n_reg):
    url = f"https://dl.fbaipublicfiles.com/dinov2/{name}/{name}{'_reg4' if n_reg else ''}_pretrain.pth"
    sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    pe = sd["pos_embed"]
    grid, s = _resize_pos(pe[0, 1:], vit.grid)
    sd["pos_embed"] = torch.cat([pe[:, :1], grid[None].to(pe.dtype)], 1)
    sd.pop("mask_token", None)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    print(f"Loaded {url} | pos_embed {s}x{s} -> {vit.grid[0]}x{vit.grid[1]}")

# ------------------------------ CLIP ViT (key khớp HuggingFace CLIPVisionModel + visual_projection) ------------------------------
class _QuickGELU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(1.702 * x)

class _ClipAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads
        self.q_proj = nn.Linear(dim, dim); self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim); self.out_proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape; h = self.h
        q = self.q_proj(x).view(B, N, h, C // h).transpose(1, 2)
        k = self.k_proj(x).view(B, N, h, C // h).transpose(1, 2)
        v = self.v_proj(x).view(B, N, h, C // h).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(x.transpose(1, 2).reshape(B, N, C))

class _ClipMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.fc1 = nn.Linear(dim, hidden); self.act = _QuickGELU(); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class _ClipLayer(nn.Module):
    def __init__(self, dim, heads, dp):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-5); self.self_attn = _ClipAttn(dim, heads)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-5); self.mlp = _ClipMlp(dim, dim * 4)
        self.dp = dp
    def forward(self, x):
        x = x + _drop_path(self.self_attn(self.layer_norm1(x)), self.dp, self.training)
        return x + _drop_path(self.mlp(self.layer_norm2(x)), self.dp, self.training)

class _ClipEmb(nn.Module):
    def __init__(self, dim, patch, grid):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.zeros(dim))
        self.patch_embedding = nn.Conv2d(3, dim, patch, patch, bias=False)
        self.position_embedding = nn.Embedding(1 + grid[0] * grid[1], dim)

class _ClipEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dps):
        super().__init__(); self.layers = nn.ModuleList([_ClipLayer(dim, heads, d) for d in dps])

class ClipViT(nn.Module):
    def __init__(self, dim, depth, heads, grid, patch, proj_dim, drop_path=0.0, parts=False):
        super().__init__()
        self.grid, self.dim, self.proj_dim, self.parts = grid, dim, proj_dim, parts
        self.embeddings = _ClipEmb(dim, patch, grid)
        self.pre_layrnorm = nn.LayerNorm(dim, eps=1e-5)          # tên giống hệt HF (kể cả lỗi chính tả)
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.encoder = _ClipEncoder(dim, depth, heads, dps)
        self.post_layernorm = nn.LayerNorm(dim, eps=1e-5)
        self.visual_projection = nn.Linear(dim, proj_dim, bias=False)
        self.out_dim = proj_dim + (1 + 3 * parts) * dim
    @property
    def blocks(self): return self.encoder.layers
    def embed(self, x):
        B = x.shape[0]
        p = self.embeddings.patch_embedding(x).flatten(2).transpose(1, 2)
        c = self.embeddings.class_embedding.to(p.dtype).expand(B, 1, -1)
        x = torch.cat([c, p], 1) + self.embeddings.position_embedding.weight[None].to(p.dtype)
        x = self.pre_layrnorm(x)
        for l in self.encoder.layers: x = l(x)
        return x
    def image_embeds(self, x):                                  # giống CLIPModel.get_image_features
        return self.visual_projection(self.post_layernorm(self.embed(x)[:, 0]))
    def forward(self, x, return_tokens=False):
        x = self.embed(x)
        e = F.normalize(self.visual_projection(self.post_layernorm(x[:, 0])).float(), dim=-1)   # embedding ảnh CLIP
        f = [e, self.post_layernorm(x[:, 1:].mean(1)).float()]                                  # trung bình patch
        if self.parts: f += [self.post_layernorm(s).float() for s in _stripe_means(x[:, 1:], self.grid)]
        f = torch.cat(f, 1)
        return (f, self.post_layernorm(x[:, 1:])) if return_tokens else f

CLIP_CFG = {"clip_vitb16": ("openai/clip-vit-base-patch16", 768, 12, 12, 16, 512),
            "clip_vitl14": ("openai/clip-vit-large-patch14", 1024, 24, 16, 14, 768)}

def clip_state_from_hf(hf_sd, grid):
    sd = {k[len("vision_model."):]: v for k, v in hf_sd.items() if k.startswith("vision_model.")}
    sd["visual_projection.weight"] = hf_sd["visual_projection.weight"]
    sd.pop("embeddings.position_ids", None)
    pe = sd["embeddings.position_embedding.weight"]
    g, s = _resize_pos(pe[1:], grid)
    sd["embeddings.position_embedding.weight"] = torch.cat([pe[:1].float(), g], 0).to(pe.dtype)
    return sd, s

def load_clip_pretrained(vit, arch):
    from transformers import CLIPModel            # chỉ cần khi train, run.py không dùng
    hf = CLIPModel.from_pretrained(CLIP_CFG[arch][0]).state_dict()
    sd, s = clip_state_from_hf(hf, vit.grid)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    print(f"Loaded {CLIP_CFG[arch][0]} | pos_embed {s}x{s} -> {vit.grid[0]}x{vit.grid[1]}")

# ------------------------------ SigLIP / SigLIP 2 ViT (key khớp HuggingFace SiglipVisionModel) ------------------------------
class _GELUTanh(nn.Module):
    def forward(self, x): return F.gelu(x, approximate="tanh")

class _SigMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.fc1 = nn.Linear(dim, hidden); self.act = _GELUTanh(); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class _SigLayer(nn.Module):
    def __init__(self, dim, heads, dp, hidden=None):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6); self.self_attn = _ClipAttn(dim, heads)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6); self.mlp = _SigMlp(dim, hidden or dim * 4)
        self.dp = dp
    def forward(self, x):
        x = x + _drop_path(self.self_attn(self.layer_norm1(x)), self.dp, self.training)
        return x + _drop_path(self.mlp(self.layer_norm2(x)), self.dp, self.training)

class _SigEmb(nn.Module):
    def __init__(self, dim, patch, grid):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, dim, patch, patch)
        self.position_embedding = nn.Embedding(grid[0] * grid[1], dim)

class _SigHead(nn.Module):                      # multihead attention pooling (MAP)
    def __init__(self, dim, heads, hidden=None):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, dim))
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.layernorm = nn.LayerNorm(dim, eps=1e-6); self.mlp = _SigMlp(dim, hidden or dim * 4)
    def forward(self, x):
        h = self.attention(self.probe.expand(x.shape[0], -1, -1).to(x.dtype), x, x, need_weights=False)[0]
        h = h + self.mlp(self.layernorm(h))
        return h[:, 0]

class _SigEncoder(nn.Module):
    def __init__(self, dim, heads, dps, hidden=None):
        super().__init__(); self.layers = nn.ModuleList([_SigLayer(dim, heads, d, hidden) for d in dps])

class SiglipViT(nn.Module):
    def __init__(self, dim, depth, heads, grid, patch, drop_path=0.0, parts=False, hidden=None):
        super().__init__()
        self.grid, self.dim, self.parts = grid, dim, parts
        self.embeddings = _SigEmb(dim, patch, grid)
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.encoder = _SigEncoder(dim, heads, dps, hidden)
        self.post_layernorm = nn.LayerNorm(dim, eps=1e-6)
        self.head = _SigHead(dim, heads, hidden)
        self.out_dim = (2 + 3 * parts) * dim
    @property
    def blocks(self): return self.encoder.layers
    def embed(self, x):
        p = self.embeddings.patch_embedding(x).flatten(2).transpose(1, 2)
        x = p + self.embeddings.position_embedding.weight[None].to(p.dtype)
        for l in self.encoder.layers: x = l(x)
        return self.post_layernorm(x)
    def image_embeds(self, x):                      # giống SiglipModel.get_image_features
        return self.head(self.embed(x))
    def forward(self, x, return_tokens=False):
        x = self.embed(x)
        e = F.normalize(self.head(x).float(), dim=-1)
        f = torch.cat([e, x.mean(1).float()] + ([s.float() for s in _stripe_means(x, self.grid)] if self.parts else []), 1)
        return (f, x) if return_tokens else f

SIGLIP_CFG = {"siglip2_vitb16": ("google/siglip2-base-patch16-224", 768, 12, 12, 16),
              "siglip_vitb16":  ("google/siglip-base-patch16-224", 768, 12, 12, 16),
              # 11/10 G14: SigLIP2 So400m/16 (Apache-2.0; 27 khối, dim 1152, MLP 4304; pos_embed 16×16 nội suy về lưới ảnh). Thành viên thứ 2 khác họ DINO
              "siglip2_so400m16": ("google/siglip2-so400m-patch16-256", 1152, 27, 16, 16, 4304),
              "siglip2_dbg": (None, 64, 2, 2, 16, 96)}                    # chỉ để chạy thử code trên CPU

def siglip_state_from_hf(hf_sd, grid):
    sd = {k[len("vision_model."):]: v for k, v in hf_sd.items() if k.startswith("vision_model.")}
    sd.pop("embeddings.position_ids", None)
    pe = sd["embeddings.position_embedding.weight"]
    g, s = _resize_pos(pe, grid, antialias=False)        # giống interpolate_pos_encoding của HF
    sd["embeddings.position_embedding.weight"] = g.to(pe.dtype)
    return sd, s

def load_siglip_pretrained(vit, arch):
    from transformers import AutoModel
    hf = AutoModel.from_pretrained(SIGLIP_CFG[arch][0]).state_dict()
    sd, s = siglip_state_from_hf(hf, vit.grid)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    print(f"Loaded {SIGLIP_CFG[arch][0]} | pos_embed {s}x{s} -> {vit.grid[0]}x{vit.grid[1]}")

# ------------------------------ Decoder truy vấn thuộc tính (K11) ------------------------------
class _XAttn(nn.Module):
    # attention đa đầu viết bằng nn.Linear thường (quantize_dynamic xử lý được, không dùng nn.MultiheadAttention)
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim); self.o = nn.Linear(dim, dim)
    def forward(self, q, kv):
        B, Nq, C = q.shape; Nk = kv.shape[1]; h = self.h
        q = self.q(q).view(B, Nq, h, C // h).transpose(1, 2)
        k = self.k(kv).view(B, Nk, h, C // h).transpose(1, 2)
        v = self.v(kv).view(B, Nk, h, C // h).transpose(1, 2)
        return self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Nq, C))

class QueryDecoder(nn.Module):
    # Mỗi thuộc tính 1 query: cross-attention lên patch token (tự tìm vùng ảnh liên quan), rồi self-attention giữa 40 query
    # (quan hệ giữa thuộc tính, như một đồ thị học được theo từng ảnh). Đầu ra cộng vào logit của head tuyến tính.
    # w = 0 lúc khởi tạo nên ban đầu model y hệt bản gốc.
    def __init__(self, dim, num_attr, heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_attr, dim) * 0.02)
        self.ln_q = nn.LayerNorm(dim); self.ln_kv = nn.LayerNorm(dim); self.cross = _XAttn(dim, heads)
        self.ln_s = nn.LayerNorm(dim); self.self_attn = _XAttn(dim, heads)
        self.ln_f = nn.LayerNorm(dim); self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.w = nn.Parameter(torch.zeros(num_attr, dim)); self.b = nn.Parameter(torch.zeros(num_attr))
    def forward(self, tok):
        kv = self.ln_kv(tok)
        h = self.query.expand(tok.shape[0], -1, -1)
        h = h + self.cross(self.ln_q(h), kv)
        s = self.ln_s(h); h = h + self.self_attn(s, s)
        h = h + self.mlp(self.ln_f(h))
        return (h.float() * self.w).sum(-1) + self.b

# ------------------------------ DINOv3 ViT (RoPE 2D, patch 16), khoá khớp transformers DINOv3ViTModel ------------------------------
DINO3_CFG = {"dinov3_vits16": (384, 12, 6, "facebook/dinov3-vits16-pretrain-lvd1689m"),
             "dinov3_vitb16": (768, 12, 12, "facebook/dinov3-vitb16-pretrain-lvd1689m"),
             "dinov3_vitl16": (1024, 24, 16, "facebook/dinov3-vitl16-pretrain-lvd1689m")}

def _rot_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def _v3_rope(gh, gw, head_dim, base, device):
    # cos / sin cho từng patch: toạ độ tâm patch đưa về [-1, 1] (như get_patches_center_coordinates, không augment khi train)
    ch = torch.arange(0.5, gh, dtype=torch.float32, device=device) / gh
    cw = torch.arange(0.5, gw, dtype=torch.float32, device=device) / gw
    coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), -1).flatten(0, 1) * 2.0 - 1.0
    inv = 1.0 / base ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32, device=device)
    ang = (2 * math.pi * coords[:, :, None] * inv[None, None, :]).flatten(1, 2).tile(2)
    return ang.cos(), ang.sin()

class _V3Emb(nn.Module):
    def __init__(self, dim, n_reg, patch):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))         # không dùng, chỉ để khớp checkpoint
        self.register_tokens = nn.Parameter(torch.zeros(1, n_reg, dim))
        self.patch_embeddings = nn.Conv2d(3, dim, patch, patch)

class _V3Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads
        self.q_proj = nn.Linear(dim, dim); self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim); self.o_proj = nn.Linear(dim, dim)
    def forward(self, x, cos, sin, n_pre):
        B, N, C = x.shape; hd = C // self.h
        q, k, v = (f(x).view(B, N, self.h, hd).transpose(1, 2) for f in (self.q_proj, self.k_proj, self.v_proj))
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        rope = lambda t: torch.cat([t[:, :, :n_pre], t[:, :, n_pre:] * cos + _rot_half(t[:, :, n_pre:]) * sin], 2)   # RoPE chỉ cho patch
        o = F.scaled_dot_product_attention(rope(q), rope(k), v)
        return self.o_proj(o.transpose(1, 2).reshape(B, N, C))

class _V3LS(nn.Module):
    def __init__(self, dim): super().__init__(); self.lambda1 = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * self.lambda1

class _V3Mlp(nn.Module):
    def __init__(self, dim, hidden): super().__init__(); self.up_proj = nn.Linear(dim, hidden); self.down_proj = nn.Linear(hidden, dim)
    def forward(self, x): return self.down_proj(F.gelu(self.up_proj(x)))

class _V3Layer(nn.Module):
    def __init__(self, dim, heads, dp, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps); self.attention = _V3Attn(dim, heads); self.layer_scale1 = _V3LS(dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps); self.mlp = _V3Mlp(dim, dim * 4); self.layer_scale2 = _V3LS(dim)
        self.dp = dp
    def forward(self, x, cos, sin, n_pre):
        x = x + _drop_path(self.layer_scale1(self.attention(self.norm1(x), cos, sin, n_pre)), self.dp, self.training)
        return x + _drop_path(self.layer_scale2(self.mlp(self.norm2(x))), self.dp, self.training)

class DinoV3ViT(nn.Module):
    # Cùng giao diện với DinoViT (features / forward / grid / out_dim / blocks); RoPE nên đổi kích thước ảnh không cần nội suy gì
    def __init__(self, dim, depth, heads, grid, n_reg=4, drop_path=0.0, parts=False, patch=16, eps=1e-5, theta=100.0):
        super().__init__()
        self.grid, self.n_reg, self.parts, self.patch, self.heads, self.theta = grid, n_reg, parts, patch, heads, theta
        self.embeddings = _V3Emb(dim, n_reg, patch)
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.layer = nn.ModuleList([_V3Layer(dim, heads, d, eps) for d in dps])
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.out_dim = (2 + 3 * parts) * dim
        self._rope = {}
    @property
    def blocks(self): return self.layer
    def features(self, x, taps=()):
        B, _, H, W = x.shape; self.grid = (H // self.patch, W // self.patch)
        key = (self.grid, str(x.device))
        if key not in self._rope: self._rope[key] = _v3_rope(*self.grid, self.norm.normalized_shape[0] // self.heads, self.theta, x.device)
        cos, sin = self._rope[key]; e = self.embeddings; n_pre = 1 + self.n_reg
        x = torch.cat([e.cls_token.expand(B, -1, -1), e.register_tokens.expand(B, -1, -1),
                       e.patch_embeddings(x).flatten(2).transpose(1, 2)], 1)
        mids = []
        for i, L_ in enumerate(self.layer):
            x = L_(x, cos, sin, n_pre)
            if i in taps: mids.append(x[:, n_pre:])
        x = self.norm(x); pt = x[:, n_pre:]
        f = torch.cat([x[:, 0], pt.mean(1)] + (_stripe_means(pt, self.grid) if self.parts else []), 1)
        return f, pt, mids
    def forward(self, x, return_tokens=False):
        f, pt, _ = self.features(x)
        return (f, pt) if return_tokens else f

def load_dinov3_pretrained(vit, repo):
    # Chỉ dùng lúc TRAIN (cần mạng, quyền tải repo gated, token trong biến môi trường HF_TOKEN). Server không gọi hàm này.
    import os
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download(repo, "model.safetensors", token=os.environ.get("HF_TOKEN") or None))
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, ("DINOv3 khoá không khớp", missing[:5], unexpected[:5])
    print(f"Loaded {repo} (DINOv3, {len(sd)} tensor)")

# ------------------------------ DINOv3 ViT-H+/16 (840M), khoá khớp timm vit_huge_plus_patch16_dinov3 (repo timm không gated) ------------------------------
# Như DinoV3ViT nhưng: qkv gộp không bias, layer scale gamma_1/2, MLP SwiGLU (fc1_g, fc1_x, fc2; ẩn 5120), tên khoá kiểu timm Eva.
# RoPE giống hệt DINOv3-L (theta 100, rotate-half, chỉ cho patch). Thứ tự token: cls, 4 register, patch.
DINO3_EVA_CFG = {"dinov3_vith16p": (1280, 32, 20, 5120, "timm/vit_huge_plus_patch16_dinov3.lvd1689m"),   # dim, depth, heads, ẩn MLP, repo
                 "dinov3_dbgeva": (64, 3, 2, 128, None)}                                                  # chỉ để chạy thử code trên CPU

class _EvaAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads; self.qkv = nn.Linear(dim, dim * 3, bias=False); self.proj = nn.Linear(dim, dim)
    def forward(self, x, cos, sin, n_pre):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        rope = lambda t: torch.cat([t[:, :, :n_pre], t[:, :, n_pre:] * cos + _rot_half(t[:, :, n_pre:]) * sin], 2)
        o = F.scaled_dot_product_attention(rope(q), rope(k), v)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))

class _SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.fc1_g = nn.Linear(dim, hidden); self.fc1_x = nn.Linear(dim, hidden); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(F.silu(self.fc1_g(x)) * self.fc1_x(x))

class _EvaBlock(nn.Module):
    def __init__(self, dim, heads, hidden, dp, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps); self.attn = _EvaAttn(dim, heads); self.gamma_1 = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim, eps=eps); self.mlp = _SwiGLU(dim, hidden); self.gamma_2 = nn.Parameter(torch.ones(dim))
        self.dp = dp
    def forward(self, x, cos, sin, n_pre):
        x = x + _drop_path(self.gamma_1 * self.attn(self.norm1(x), cos, sin, n_pre), self.dp, self.training)
        return x + _drop_path(self.gamma_2 * self.mlp(self.norm2(x)), self.dp, self.training)

class DinoV3Eva(nn.Module):
    # Cùng giao diện với DinoV3ViT (features / forward / grid / out_dim / blocks / norm). ckpt=True: gradient checkpointing cho khối còn train
    def __init__(self, dim, depth, heads, hidden, grid, n_reg=4, drop_path=0.0, parts=False, patch=16, eps=1e-5, theta=100.0):
        super().__init__()
        self.grid, self.n_reg, self.parts, self.patch, self.heads, self.theta, self.ckpt = grid, n_reg, parts, patch, heads, theta, False
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)); self.reg_token = nn.Parameter(torch.zeros(1, n_reg, dim))
        self.patch_embed = nn.Module(); self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([_EvaBlock(dim, heads, hidden, d, eps) for d in dps])
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.out_dim = (2 + 3 * parts) * dim
        self._rope = {}
    def tokens(self, x, taps=()):
        B, _, H, W = x.shape; self.grid = (H // self.patch, W // self.patch)
        key = (self.grid, str(x.device))
        if key not in self._rope: self._rope[key] = _v3_rope(*self.grid, self.norm.normalized_shape[0] // self.heads, self.theta, x.device)
        cos, sin = self._rope[key]; n_pre = 1 + self.n_reg
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.reg_token.expand(B, -1, -1),
                       self.patch_embed.proj(x).flatten(2).transpose(1, 2)], 1)
        mids = []
        for i, b in enumerate(self.blocks):
            if self.ckpt and self.training and torch.is_grad_enabled() and any(p_.requires_grad for p_ in b.parameters()):
                x = torch.utils.checkpoint.checkpoint(b, x, cos, sin, n_pre, use_reentrant=False)
            else: x = b(x, cos, sin, n_pre)
            if i in taps: mids.append(x[:, n_pre:])
        return self.norm(x), n_pre, mids
    def features(self, x, taps=()):
        x, n_pre, mids = self.tokens(x, taps); pt = x[:, n_pre:]
        f = torch.cat([x[:, 0], pt.mean(1)] + (_stripe_means(pt, self.grid) if self.parts else []), 1)
        return f, pt, mids
    def forward(self, x, return_tokens=False):
        f, pt, _ = self.features(x)
        return (f, pt) if return_tokens else f

_EVA_CHECKED = set()
def load_dinov3_eva_pretrained(vit, repo, timm_name=None):
    # Chỉ dùng lúc TRAIN (cần mạng; repo timm không gated). Lần nạp đầu: so đầu ra với timm (nếu có) trên ảnh ngẫu nhiên, fp32 CPU.
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download(repo, "model.safetensors"))
    sd = {k: v for k, v in sd.items() if not k.startswith(("head.", "fc_norm.", "rope."))}
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, ("DINOv3 Eva khoá không khớp", missing[:5], unexpected[:5])
    print(f"Loaded {repo} (DINOv3 Eva, {len(sd)} tensor, {sum(v.numel() for v in sd.values()) / 1e6:.1f}M)")
    if repo in _EVA_CHECKED: return
    _EVA_CHECKED.add(repo)
    try:
        import timm
        tm = timm.create_model(timm_name or repo.split("/")[-1].split(".")[0], pretrained=False, num_classes=0, dynamic_img_size=True)
        tm.load_state_dict(sd, strict=False); tm.eval()
        g0, was = vit.grid, vit.training; vit.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 256, 128)
            ref = tm.forward_features(x); ours, n_pre, _ = vit.tokens(x)
        vit.grid = g0; vit.train(was)
        err = float((ref - ours).abs().max()); rel = err / float(ref.abs().max())
        print(f"[DINOv3 Eva] so với timm {timm.__version__}: lệch tuyệt đối max {err:.2e}, tương đối {rel:.2e}")
        assert rel < 1e-3, "DINOv3 Eva KHÔNG khớp timm"
        del tm
    except AssertionError: raise
    except Exception as e:
        print(f"[DINOv3 Eva] bỏ qua so sánh timm ({type(e).__name__}: {e})")

# ------------------------------ C-RADIOv4 (NVIDIA, 01/2026; NVIDIA Open Model License, weights HF không gated) ------------------------------
# 11/10: backbone chưng cất từ SigLIP2-g + DINOv3-7B + SAM3 (IN kNN 86.6 > DINOv3-H+ 85.8). Kiến trúc = timm vit_huge_patch16 (qkv có bias, MLP GELU,
# LN eps 1e-6, không layer scale, KHÔNG có norm cuối) + CPE: pos_embed học được trên lưới 128×128; khi suy luận nội suy song tuyến về
# (max(h,w))² rồi cắt góc trên-trái h×w (đúng ViTPatchGenerator._get_pos_embeddings, nhánh eval). Patch embed = Linear (c, yy, xx) không bias.
# 10 token đầu = 3 CLS theo teacher + 7 register; đặc trưng tóm tắt = CLS ở summary_idxs (2 teacher có summary). Đầu vào chuẩn hoá CLIP OpenAI.
RADIO_CFG = {"radio_v4h": (1280, 32, 16, 5120, 10, 128, "nvidia/C-RADIOv4-H"),      # dim, depth, heads, ẩn MLP, số token đầu, lưới pos, repo
             "radio_dbg": (64, 3, 2, 128, 10, 16, None)}                              # chỉ để chạy thử code trên CPU

class _VitAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads; self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))

class _VitMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.fc1 = nn.Linear(dim, hidden); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))

class _VitBlock(nn.Module):
    def __init__(self, dim, heads, hidden, dp, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps); self.attn = _VitAttn(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps); self.mlp = _VitMlp(dim, hidden); self.dp = dp
    def forward(self, x):
        x = x + _drop_path(self.attn(self.norm1(x)), self.dp, self.training)
        return x + _drop_path(self.mlp(self.norm2(x)), self.dp, self.training)

class RadioViT(nn.Module):
    # Cùng giao diện với DinoV3Eva (tokens / features / forward / grid / out_dim / blocks / norm / ckpt).
    def __init__(self, dim, depth, heads, hidden, n_pre, pos_grid, grid, drop_path=0.0, parts=False, patch=16, eps=1e-6, n_sum=2):
        super().__init__()
        self.grid, self.n_pre, self.parts, self.patch, self.pos_grid, self.ckpt = grid, n_pre, parts, patch, pos_grid, False
        self.cls_token = nn.Parameter(torch.zeros(n_pre, dim))                      # 3 CLS theo teacher + 7 register
        self.patch_embed = nn.Linear(3 * patch * patch, dim, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_grid * pos_grid, dim))
        self.register_buffer("summary_idxs", torch.arange(n_sum, dtype=torch.long))
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([_VitBlock(dim, heads, hidden, d, eps) for d in dps])
        self.norm = nn.Identity()
        self.out_dim = (n_sum + 1 + 3 * parts) * dim
        self._pos = {}
    def pos(self, gh, gw, dtype):
        # nhánh eval của CPE; khi train tính lại mỗi lần (pos_embed còn học), khi eval lưu đệm theo lưới
        key = (gh, gw, str(self.pos_embed.device), dtype)
        if not self.training and key in self._pos: return self._pos[key]
        P, C = self.pos_grid, self.pos_embed.shape[-1]; m = max(gh, gw)
        pe = self.pos_embed.reshape(1, P, P, C).permute(0, 3, 1, 2)
        if (gh, gw) != (P, P):
            pe = F.interpolate(pe.float(), size=(m, m), mode="bilinear", align_corners=False)[..., :gh, :gw]
        pe = pe.flatten(2).transpose(1, 2).to(dtype)
        if not self.training: self._pos[key] = pe
        else: self._pos.clear()
        return pe
    def tokens(self, x, taps=()):
        B, _, H, W = x.shape; p = self.patch; self.grid = (H // p, W // p)
        t = F.conv2d(x, self.patch_embed.weight.reshape(-1, 3, p, p).to(x.dtype), stride=p).flatten(2).transpose(1, 2)
        t = t + self.pos(*self.grid, t.dtype)
        x = torch.cat([self.cls_token.unsqueeze(0).expand(B, -1, -1).to(t.dtype), t], 1)
        mids = []
        for i, b in enumerate(self.blocks):
            if self.ckpt and self.training and torch.is_grad_enabled() and any(p_.requires_grad for p_ in b.parameters()):
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            else: x = b(x)
            if i in taps: mids.append(x[:, self.n_pre:])
        return self.norm(x), self.n_pre, mids
    def features(self, x, taps=()):
        x, n_pre, mids = self.tokens(x, taps); pt = x[:, n_pre:]
        f = torch.cat([x[:, self.summary_idxs].flatten(1), pt.mean(1)] + (_stripe_means(pt, self.grid) if self.parts else []), 1)
        return f, pt, mids
    def forward(self, x, return_tokens=False):
        f, pt, _ = self.features(x)
        return (f, pt) if return_tokens else f

_RADIO_CHECKED = set()
def load_radio_pretrained(vit, repo):
    # Chỉ dùng lúc TRAIN (cần mạng; repo không gated). Khoá HF: radio_model.model.{blocks.*, patch_generator.{cls_token.token, embedder.weight, pos_embed}},
    # radio_model.summary_idxs, radio_model.input_conditioner.{norm_mean, norm_std}. Lần nạp đầu: so với code gốc NVIDIA (torch.hub) nếu tải được.
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    raw = load_file(hf_hub_download(repo, "model.safetensors")); sd = {}
    for k, v in raw.items():
        if k.startswith("radio_model.model.blocks."): sd[k[len("radio_model.model."):]] = v
        elif k == "radio_model.model.patch_generator.cls_token.token": sd["cls_token"] = v
        elif k == "radio_model.model.patch_generator.embedder.weight": sd["patch_embed.weight"] = v
        elif k == "radio_model.model.patch_generator.pos_embed": sd["pos_embed"] = v
        elif k == "radio_model.summary_idxs": sd["summary_idxs"] = v.long()
    mean, std = raw["radio_model.input_conditioner.norm_mean"].flatten(), raw["radio_model.input_conditioner.norm_std"].flatten()
    assert torch.allclose(mean, torch.tensor(CLIP_NORM[0]), atol=1e-4) and torch.allclose(std, torch.tensor(CLIP_NORM[1]), atol=1e-4), ("RADIO chuẩn hoá khác CLIP", mean, std)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, ("RADIO khoá không khớp", missing[:5], unexpected[:5])
    print(f"Loaded {repo} (C-RADIO, {len(sd)} tensor, {sum(v.numel() for v in sd.values()) / 1e6:.1f}M, summary_idxs {sd['summary_idxs'].tolist()})")
    if repo in _RADIO_CHECKED: return
    _RADIO_CHECKED.add(repo)
    try:   # code gốc qua torch.hub (cần timm + einops; bản transformers đòi thêm open_clip). Đã kiểm local 11/10: lệch ≤ 1.1e-5 ở 256/224/192
        ref = torch.hub.load("NVlabs/RADIO", "radio_model", version="c-radio_v4-h", progress=False, skip_validation=True, trust_repo=True)
        ref.load_state_dict({k[len("radio_model."):]: v for k, v in raw.items() if k.startswith("radio_model.")}, strict=False); ref = ref.float().eval()
        g0, was = vit.grid, vit.training; vit.eval()
        with torch.no_grad():
            x = torch.rand(1, 3, 256, 128)
            out = ref(x); summ, feat = (out.summary, out.features) if hasattr(out, "summary") else (out[0], out[1])
            mn, sd_ = torch.tensor(CLIP_NORM[0]).view(1, 3, 1, 1), torch.tensor(CLIP_NORM[1]).view(1, 3, 1, 1)
            tok, n_pre, _ = vit.tokens((x - mn) / sd_)
            ours_s, ours_f = tok[:, vit.summary_idxs].flatten(1), tok[:, n_pre:]
        vit.grid = g0; vit.train(was)
        rel = max(float((summ - ours_s).abs().max() / summ.abs().max()), float((feat - ours_f).abs().max() / feat.abs().max()))
        print(f"[C-RADIO] so với code NVIDIA: lệch tương đối max {rel:.2e}")
        assert rel < 1e-3, "C-RADIO KHÔNG khớp code gốc"
        del ref
    except AssertionError: raise
    except Exception as e:
        print(f"[C-RADIO] bỏ qua so sánh code gốc ({type(e).__name__}: {e})")

# ------------------------------ PE-Core (Meta Perception Encoder, Apache-2.0), khoá khớp timm vit_pe_core_* ------------------------------
# Như timm Eva cấu hình PE: cls + pos_embed tuyệt đối + norm_pre, khối qkv gộp với RoPE 2D kiểu "cat" (xen kẽ, chỉ cho patch, lưới lệch 1,
# quy về lưới pretrain 14×14), MLP GELU, không layer scale; norm cuối; pooling bằng attention (1 latent, 8 head). Patch embed không bias.
PE_CFG = {"pe_core_b16": ("timm/vit_pe_core_base_patch16_224.fb", 768, 12, 12, 16, 14),    # repo, dim, depth, heads, patch, lưới pretrain
          "pe_core_l14": ("timm/vit_pe_core_large_patch14_336.fb", 1024, 24, 16, 14, 24)}
PE_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def _pe_rope(gh, gw, head_dim, ref, device, temp=10000.0, offset=1.0):
    # = timm build_rotary_pos_embed(in_pixels=False, grid_indexing="xy", grid_offset=1, ref_feat_shape=(ref, ref)) -> sin, cos (N, head_dim)
    nb = head_dim // 4
    bands = 1.0 / temp ** (torch.arange(nb, dtype=torch.float32, device=device) / nb)
    tx = (torch.arange(gw, dtype=torch.float32, device=device) + offset) / gw * ref
    ty = (torch.arange(gh, dtype=torch.float32, device=device) + offset) / gh * ref
    gx, gy = torch.meshgrid(tx, ty, indexing="xy")
    pos = torch.stack([gx, gy], -1).unsqueeze(-1) * bands
    return pos.sin().reshape(gh * gw, -1).repeat_interleave(2, -1), pos.cos().reshape(gh * gw, -1).repeat_interleave(2, -1)

def _pe_rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)

class _PEAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads; self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)
    def forward(self, x, sin, cos):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        sin, cos = sin.to(q.dtype), cos.to(q.dtype)
        rope = lambda t: torch.cat([t[:, :, :1], t[:, :, 1:] * cos + _pe_rot(t[:, :, 1:]) * sin], 2)   # RoPE chỉ cho patch (bỏ cls)
        o = F.scaled_dot_product_attention(rope(q), rope(k), v)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))

class _PEMlp(nn.Module):
    def __init__(self, dim, hidden): super().__init__(); self.fc1 = nn.Linear(dim, hidden); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))

class _PEBlock(nn.Module):
    def __init__(self, dim, heads, dp):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5); self.attn = _PEAttn(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5); self.mlp = _PEMlp(dim, dim * 4); self.dp = dp
    def forward(self, x, sin, cos):
        x = x + _drop_path(self.attn(self.norm1(x), sin, cos), self.dp, self.training)
        return x + _drop_path(self.mlp(self.norm2(x)), self.dp, self.training)

class _PEPool(nn.Module):
    # = timm AttentionPoolLatent (1 latent, pool "token")
    def __init__(self, dim, heads=8):
        super().__init__(); self.h = heads
        self.latent = nn.Parameter(torch.zeros(1, 1, dim)); self.q = nn.Linear(dim, dim); self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim); self.norm = nn.LayerNorm(dim, eps=1e-5); self.mlp = _PEMlp(dim, dim * 4)
    def forward(self, x):
        B, N, C = x.shape; hd = C // self.h
        q = self.q(self.latent.expand(B, -1, -1)).reshape(B, 1, self.h, hd).transpose(1, 2)
        k, v = self.kv(x).reshape(B, N, 2, self.h, hd).permute(2, 0, 3, 1, 4).unbind(0)
        o = self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, 1, C))
        o = o + self.mlp(self.norm(o))
        return o[:, 0]

class _PEPatch(nn.Module):
    def __init__(self, dim, patch): super().__init__(); self.proj = nn.Conv2d(3, dim, patch, patch, bias=False)
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)

class PECoreViT(nn.Module):
    # Cùng giao diện với DinoV3ViT (features / forward / grid / out_dim / blocks / norm). Đặc trưng = [pooling attention ; TB patch].
    # pos_embed tuyệt đối theo lưới lúc tạo model -> mọi ảnh phải đúng kích thước đó (pipeline luôn resize về IMG).
    def __init__(self, dim, depth, heads, grid, patch=16, ref=14, drop_path=0.0, parts=False):
        super().__init__()
        self.grid, self.patch, self.heads, self.ref, self.parts = tuple(grid), patch, heads, ref, parts
        self.patch_embed = _PEPatch(dim, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid[0] * grid[1], dim))
        self.norm_pre = nn.LayerNorm(dim, eps=1e-5)
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([_PEBlock(dim, heads, dps[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-5); self.attn_pool = _PEPool(dim)
        self.out_dim = dim * (2 + (3 if parts else 0)); self._rope = {}
    def features(self, x, taps=()):
        B, _, H, W = x.shape; g = (H // self.patch, W // self.patch)
        assert g == self.grid, f"PE-Core: lưới ảnh {g} khác lưới lúc tạo model {self.grid}"
        key = str(x.device)
        if key not in self._rope: self._rope[key] = _pe_rope(*g, self.pos_embed.shape[-1] // self.heads, self.ref, x.device)
        sin, cos = self._rope[key]
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.patch_embed(x)], 1) + self.pos_embed
        x = self.norm_pre(x); mids = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, sin, cos)
            if i in taps: mids.append(x[:, 1:])
        x = self.norm(x); pt = x[:, 1:]
        f = torch.cat([self.attn_pool(x), pt.mean(1)] + (_stripe_means(pt, self.grid) if self.parts else []), 1)
        return f, pt, mids
    def forward(self, x, return_tokens=False):
        f, pt, _ = self.features(x)
        return (f, pt) if return_tokens else f

def load_pe_pretrained(vit, repo):
    # Chỉ dùng lúc TRAIN (cần mạng). pos_embed pretrain (lưới 14×14) nội suy bicubic + antialias sang lưới hiện tại như timm resample_abs_pos_embed.
    import os
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download(repo, "model.safetensors", token=os.environ.get("HF_TOKEN") or None))
    sd = {k: v for k, v in sd.items() if not k.startswith(("head.", "fc_norm."))}
    pe = sd["pos_embed"]; n0 = int(round((pe.shape[1] - 1) ** 0.5)); gh, gw = vit.grid
    if (n0, n0) != (gh, gw):
        g = pe[:, 1:].reshape(1, n0, n0, -1).permute(0, 3, 1, 2).float()
        g = F.interpolate(g, size=(gh, gw), mode="bicubic", antialias=True, align_corners=False)
        sd["pos_embed"] = torch.cat([pe[:, :1].float(), g.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)], 1).to(pe.dtype)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, ("PE-Core khoá không khớp", missing[:5], unexpected[:5])
    print(f"Loaded {repo} (PE-Core, {len(sd)} tensor, pos_embed {n0}x{n0} -> {gh}x{gw})")

# ------------------------------ E12: nhánh màu ------------------------------
# Lý do: DINOv2 được pretrain với ColorJitter(hue 0.1, saturation 0.2, p 0.8) + 20 % ảnh xám, nên đặc trưng cuối được dạy để bất biến
# với màu (L tăng mọi nhóm trừ màu). Hai nhánh cộng thêm logit cho 24 thuộc tính màu, khởi tạo bằng 0 (lúc đầu = model gốc).
UB_COLOR = list(range(8, 20))       # UpperBody-Color-* theo thứ tự 40 thuộc tính của BTC
LB_COLOR = list(range(21, 33))      # LowerBody-Color-*

class MidColorHead(nn.Module):
    # Head màu từ patch token của các khối giữa (còn giữ thông tin màu), gộp theo dải thân trên / thân dưới
    def __init__(self, dim, n_taps, n_attr=40):
        super().__init__()
        self.n_attr = n_attr
        self.norm_u, self.norm_l = nn.LayerNorm(dim * n_taps), nn.LayerNorm(dim * n_taps)
        self.ub, self.lb = nn.Linear(dim * n_taps, len(UB_COLOR)), nn.Linear(dim * n_taps, len(LB_COLOR))
        for l_ in (self.ub, self.lb): nn.init.zeros_(l_.weight); nn.init.zeros_(l_.bias)
        self.register_buffer("iu", torch.tensor(UB_COLOR), persistent=False)
        self.register_buffer("il", torch.tensor(LB_COLOR), persistent=False)
    def forward(self, mids, grid):
        st = [_stripe_means(t, grid) for t in mids]
        u = self.ub(self.norm_u(torch.cat([a[1] for a in st], 1))); l_ = self.lb(self.norm_l(torch.cat([a[2] for a in st], 1)))
        return u.new_zeros(u.shape[0], self.n_attr).index_copy(1, self.iu, u).index_copy(1, self.il, l_)

class ColorStats(nn.Module):
    # Thống kê màu tường minh: histogram mềm trên lưới k×k×k màu RGB của pixel thật (đã bỏ chuẩn hoá), vùng áo và vùng quần
    # (bỏ 20 % mỗi bên để bớt nền), qua MLP nhỏ. Rẻ: ảnh được avg-pool 4×4 trước khi tính.
    def __init__(self, mean, std, k=4, hidden=128, n_attr=40):
        super().__init__()
        g = torch.linspace(0, 1, k)
        self.register_buffer("c", torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3), persistent=False)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.inv_tau, self.n_attr = 2.0 * (k - 1) ** 2, n_attr            # 1 / (2σ²), σ = nửa khoảng cách lưới
        self.mu = nn.Sequential(nn.Linear(k ** 3, hidden), nn.GELU(), nn.Linear(hidden, len(UB_COLOR)))
        self.ml = nn.Sequential(nn.Linear(k ** 3, hidden), nn.GELU(), nn.Linear(hidden, len(LB_COLOR)))
        for l_ in (self.mu[2], self.ml[2]): nn.init.zeros_(l_.weight); nn.init.zeros_(l_.bias)
        self.register_buffer("iu", torch.tensor(UB_COLOR), persistent=False)
        self.register_buffer("il", torch.tensor(LB_COLOR), persistent=False)
    def hist(self, img):
        px = img.flatten(2).transpose(1, 2)                                 # (B, P, 3)
        d2 = ((px.unsqueeze(2) - self.c) ** 2).sum(-1)                      # (B, P, k³)
        return torch.softmax(-d2 * self.inv_tau, -1).mean(1).sqrt()
    def forward(self, x):
        img = F.avg_pool2d((x.float() * self.std + self.mean).clamp(0, 1), 4)
        H, W = img.shape[-2:]; c0 = int(W * 0.2); c1 = max(int(W * 0.8), c0 + 1)
        u = self.mu(self.hist(img[:, :, int(H * 0.2):max(int(H * 0.6), int(H * 0.2) + 1), c0:c1]))
        l_ = self.ml(self.hist(img[:, :, int(H * 0.5):, c0:c1]))
        return u.new_zeros(u.shape[0], self.n_attr).index_copy(1, self.iu, u).index_copy(1, self.il, l_.to(u.dtype))

# ------------------------------ Mô hình PAR ------------------------------
def patch_of(arch):
    return (CLIP_CFG[arch][4] if arch.startswith("clip") else SIGLIP_CFG[arch][4] if arch.startswith("siglip")
            else PE_CFG[arch][4] if arch.startswith("pe_core") else 16 if arch.startswith(("dinov3", "radio")) else 14)

class DinoPartMasker(nn.Module):
    # E27 (H8): DINO đã fine-tune, đóng băng, khoanh vùng áo / quần cho model khác. Logit màu theo từng patch fc([CLS ; token_i]);
    # điểm vùng = max logit 12 màu áo / quần; mask = softmax(điểm / τ) trong khung dọc (áo 10–70 %, quần 40–100 % chiều cao).
    # fixed=True: mask đều trong cùng khung (đối chứng, không cần DINO). Ảnh vào đã chuẩn hoá theo model chính, đổi sang đầu vào của DINO.
    def __init__(self, pm, in_norm):
        super().__init__()
        self.tau, self.fixed = float(pm.get("TAU", 1.5)), bool(pm.get("FIXED", False))
        self.size = (int(pm.get("src_h", 256)), int(pm.get("src_w", 128)))
        self.net = None if self.fixed else PARNet(pm["src_arch"], pretrained=False, img_h=self.size[0], img_w=self.size[1])
        sm = pm.get("src_mean", IMAGENET_NORM[0]); ss = pm.get("src_std", IMAGENET_NORM[1])
        for n_, v in (("mi", in_norm[0]), ("si", in_norm[1]), ("md", sm), ("sd", ss)):
            self.register_buffer(n_, torch.tensor(v, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
    def train(self, mode=True):
        return super().train(False)                      # luôn ở chế độ eval (không dropout / drop path)
    @torch.no_grad()
    def forward(self, x, grid):
        B = x.shape[0]
        if self.fixed:
            GH, GW = 16, 8; su = sl = x.new_zeros(B, GH * GW, dtype=torch.float32)
        else:
            xd = F.interpolate(x.float() * self.si + self.mi, size=self.size, mode="bilinear", align_corners=False)
            fe, pt, _ = self.net.backbone.features((xd - self.md) / self.sd)
            GH, GW = self.net.backbone.grid; N, C = pt.shape[1], pt.shape[2]
            zi = self.net.fc(torch.cat([fe[:, :C].unsqueeze(1).expand(B, N, C), pt], 2)).float()
            su, sl = zi[..., UB_COLOR].amax(-1), zi[..., LB_COLOR].amax(-1)
        rows = (torch.arange(GH, device=x.device, dtype=torch.float32) / GH).view(GH, 1).expand(GH, GW).reshape(-1)
        pu = torch.where((rows >= 0.10) & (rows < 0.70), 0.0, -1e4); pl = torch.where(rows >= 0.40, 0.0, -1e4)
        def shrink(m_):
            m_ = F.adaptive_avg_pool2d(m_.view(B, 1, GH, GW), tuple(grid)).flatten(1)
            return m_ / m_.sum(1, keepdim=True).clamp(min=1e-12)
        return shrink(torch.softmax(su / self.tau + pu, 1)), shrink(torch.softmax(sl / self.tau + pl, 1))

class PartMaskColorHead(nn.Module):
    # E27: token của model chính gộp theo mask áo / quần -> logit 12 màu áo / 12 màu quần (lớp cuối khởi tạo 0 = lúc đầu như model gốc)
    def __init__(self, dim, n_src, n_attr=40):
        super().__init__()
        self.n_attr = n_attr
        self.nu, self.nl = nn.LayerNorm(dim * n_src), nn.LayerNorm(dim * n_src)
        self.ub, self.lb = nn.Linear(dim * n_src, len(UB_COLOR)), nn.Linear(dim * n_src, len(LB_COLOR))
        for l_ in (self.ub, self.lb): nn.init.zeros_(l_.weight); nn.init.zeros_(l_.bias)
        self.register_buffer("iu", torch.tensor(UB_COLOR), persistent=False)
        self.register_buffer("il", torch.tensor(LB_COLOR), persistent=False)
    def forward(self, toks, mu, ml):
        u = torch.cat([torch.einsum("bn,bnc->bc", mu.to(t.dtype), t) for t in toks], 1).float()
        l_ = torch.cat([torch.einsum("bn,bnc->bc", ml.to(t.dtype), t) for t in toks], 1).float()
        u, l_ = self.ub(self.nu(u)), self.lb(self.nl(l_))
        return u.new_zeros(u.shape[0], self.n_attr).index_copy(1, self.iu, u).index_copy(1, self.il, l_)

class PARNet(nn.Module):
    def __init__(self, arch="dinov2_vitb14_reg", num_attr=40, pretrained=False, drop=0.2,
                 img_h=252, img_w=126, drop_path=0.0, parts=False, qdec=False, cmid=(), chist=False, pmask=None):
        super().__init__()
        self.arch, self.parts = arch, parts
        self.qdec = None                                   # K11: QueryDecoder nếu qdec=True
        self.cmid, self.chead, self.cstats = tuple(cmid), None, None      # E12: nhánh màu (chỉ DINOv2)
        self.pmasker, self.phead, self.ptaps = None, None, tuple((pmask or {}).get("TAPS", ()))   # E27: màu theo vùng do DINO khoanh
        if arch.startswith(("dinov2", "dinov3", "pe_core", "radio")):
            if arch.startswith("pe_core"):                 # PE-Core: patch 16, pos_embed tuyệt đối + RoPE 2D, pooling attention
                repo, dim, depth, heads, patch, ref = PE_CFG[arch]
                self.backbone = PECoreViT(dim, depth, heads, (img_h // patch, img_w // patch), patch, ref, drop_path, parts)
                if pretrained: load_pe_pretrained(self.backbone, repo)
            elif arch in RADIO_CFG:                        # C-RADIOv4 (11/10): ViT-H/16 + CPE, 10 token đầu, chuẩn hoá CLIP
                dim, depth, heads, hidden, n_pre, pgrid, repo = RADIO_CFG[arch]
                self.backbone = RadioViT(dim, depth, heads, hidden, n_pre, pgrid, (img_h // 16, img_w // 16), drop_path, parts)
                if pretrained and repo: load_radio_pretrained(self.backbone, repo)
            elif arch in DINO3_EVA_CFG:                    # DINOv3 ViT-H+ (khoá timm)
                dim, depth, heads, hidden, repo = DINO3_EVA_CFG[arch]
                self.backbone = DinoV3Eva(dim, depth, heads, hidden, (img_h // 16, img_w // 16), 4, drop_path, parts)
                if pretrained and repo: load_dinov3_eva_pretrained(self.backbone, repo)
            elif arch.startswith("dinov3"):                # DINOv3: patch 16, RoPE, 4 register
                dim, depth, heads, repo = DINO3_CFG[arch]
                self.backbone = DinoV3ViT(dim, depth, heads, (img_h // 16, img_w // 16), 4, drop_path, parts)
                if pretrained: load_dinov3_pretrained(self.backbone, repo)
            else:
                name = arch.replace("_reg", ""); n_reg = 4 if arch.endswith("_reg") else 0
                dim, depth, heads = DINO_CFG[name]
                self.backbone = DinoViT(dim, depth, heads, (img_h // 14, img_w // 14), n_reg, 14, drop_path, parts)
                if pretrained: load_dinov2_pretrained(self.backbone, name, n_reg)
            self.fc = nn.Sequential(nn.LayerNorm(self.backbone.out_dim), nn.Dropout(drop),
                                    nn.Linear(self.backbone.out_dim, num_attr))
            if qdec: self.qdec = QueryDecoder(dim, num_attr, dim // 64)
            if cmid: self.chead = MidColorHead(dim, len(cmid), num_attr)
            if chist: self.cstats = ColorStats(*norm_for(arch), n_attr=num_attr)
            if pmask:
                assert arch.startswith("pe_core") or not self.ptaps, "E27: token khối giữa chỉ có ở PE-Core / DINO"
                self.pmasker, self.phead = DinoPartMasker(pmask, norm_for(arch)), PartMaskColorHead(dim, 1 + len(self.ptaps), num_attr)
            return
        if arch.startswith("clip"):
            _, dim, depth, heads, patch, pdim = CLIP_CFG[arch]
            self.backbone = ClipViT(dim, depth, heads, (img_h // patch, img_w // patch), patch, pdim, drop_path, parts)
            if pretrained: load_clip_pretrained(self.backbone, arch)
            # Không LayerNorm ở head: phần đầu của feature là embedding CLIP đã chuẩn hoá, head khởi tạo bằng text embedding
            self.fc = nn.Sequential(nn.Identity(), nn.Dropout(drop), nn.Linear(self.backbone.out_dim, num_attr))
            if qdec: self.qdec = QueryDecoder(dim, num_attr, dim // 64)
            if pmask:
                assert not self.ptaps, "E27: CLIP chỉ dùng token khối cuối"
                self.pmasker, self.phead = DinoPartMasker(pmask, norm_for(arch)), PartMaskColorHead(dim, 1, num_attr)
            return
        if arch.startswith("siglip"):
            _, dim, depth, heads, patch = SIGLIP_CFG[arch][:5]; hid = SIGLIP_CFG[arch][5] if len(SIGLIP_CFG[arch]) > 5 else None
            self.backbone = SiglipViT(dim, depth, heads, (img_h // patch, img_w // patch), patch, drop_path, parts, hid)
            if pretrained and SIGLIP_CFG[arch][0]: load_siglip_pretrained(self.backbone, arch)
            self.fc = nn.Sequential(nn.Identity(), nn.Dropout(drop), nn.Linear(self.backbone.out_dim, num_attr))
            if qdec: self.qdec = QueryDecoder(dim, num_attr, dim // 64)
            return
        m = getattr(torchvision.models, arch)(weights="DEFAULT" if pretrained else None)
        if arch.startswith("convnext"):
            dim = m.classifier[2].in_features
            m.classifier[2] = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, num_attr))
        elif arch.startswith("resnet"):
            dim = m.fc.in_features
            m.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, num_attr))
        else:
            raise ValueError(arch)
        self.net = m

    @property
    def head(self):
        if self.arch.startswith(("dinov2", "dinov3", "clip", "siglip", "pe_core", "radio")): return self.fc[2]
        return self.net.classifier[2][1] if self.arch.startswith("convnext") else self.net.fc[1]

    def forward(self, x, return_feat=False):
        # return_feat=True: trả thêm (đặc trưng gộp, patch token cuối) cho chưng cất đặc trưng (E11); mặc định chỉ trả logit như cũ
        if self.arch.startswith(("dinov2", "dinov3", "clip", "siglip", "pe_core", "radio")):
            mids = None
            if self.phead is not None:                     # E27: token gộp theo mask vùng của DINO
                if self.ptaps: f, tok, mids = self.backbone.features(x, self.ptaps)
                else: f, tok = self.backbone(x, return_tokens=True)
                z = self.fc(f) + self.phead([tok] + list(mids or []), *self.pmasker(x, self.backbone.grid))
                return (z, f, tok) if return_feat else z
            if self.chead is not None: f, tok, mids = self.backbone.features(x, self.cmid)
            elif self.qdec is not None or return_feat: f, tok = self.backbone(x, return_tokens=True)
            else: f, tok = self.backbone(x), None
            z = self.fc(f)
            if self.qdec is not None: z = z + self.qdec(tok)
            if self.chead is not None: z = z + self.chead(mids, self.backbone.grid)
            if self.cstats is not None: z = z + self.cstats(x)
            return (z, f, tok) if return_feat else z
        return self.net(x)
