"""
PodiumSessionValue_Pipeline.py

One-file podium-focused pipeline for the e-commerce DATATHON:
- Strong GPU-friendly **Session Transformer** to model per-session event sequences
- Optional **Tree Ensemble (CatBoost)** branch
- Robust **OOF CV + Seed/Fold ensembling + Blending**
- log1p target, early stopping, gradient clipping

USAGE (run from the folder containing train.csv, test.csv, sample_submission.csv):
    python PodiumSessionValue_Pipeline.py --epochs 30 --max_len 200 --blend 0.65

If CatBoost is not installed, the pipeline will automatically skip the tree branch and
use only the Transformer. You can install it via: pip install catboost

IMPORTANT: Adjust COLUMN_GUESS if your headers differ.
"""

import os
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------- Config & Utils ----------------------------- #

@dataclass
class Config:
    id_col: str = None          # will be inferred from sample_submission
    target_col: str = None      # will be inferred from sample_submission
    
    # event-level columns (we auto-detect names; adjust if needed)
    session_col: str = None
    user_col: Optional[str] = None
    event_type_col: Optional[str] = None
    product_col: Optional[str] = None
    category_col: Optional[str] = None
    time_col: Optional[str] = None
    price_col: Optional[str] = None

    # model
    max_len: int = 200
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    dropout: float = 0.1

    # training
    epochs: int = 30
    batch_size: int = 1024
    lr: float = 3e-4
    wd: float = 1e-4
    warmup_pct: float = 0.1
    grad_clip: float = 1.0
    
    folds: int = 5
    seeds: List[int] = None
    
    # loss
    huber_delta: float = 1.5
    use_log1p: bool = True

    # blend weight: transformer vs. catboost
    blend_weight_transformer: float = 0.65

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 77, 123, 2025]


COLUMN_GUESS = {
    'session': ['session_id', 'session', 'sess_id'],
    'user':    ['user_id', 'uid', 'customer_id'],
    'event':   ['event_type', 'event', 'action', 'type'],
    'product': ['product_id', 'item_id', 'sku', 'pid'],
    'time':    ['event_time', 'timestamp', 'time', 'ts'],
    'price':   ['price', 'value', 'amount', 'unit_price']
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------ Data preparation functions ---------------------- #

def infer_id_target(sample_path: str) -> Tuple[str, str]:
    sample = pd.read_csv(sample_path)
    assert sample.shape[1] == 2, "sample_submission.csv should have exactly two columns: id, target"
    return sample.columns[0], sample.columns[1]


def auto_detect_columns(df: pd.DataFrame, cfg: Config) -> Config:
    """Detect needed columns unless user provided them via CLI.
    CLI-provided names in cfg take precedence; otherwise try COLUMN_GUESS.
    """
    def pick(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # If user supplied explicit names, keep them. Else try to guess.
    if cfg.session_col is None:
        # Also consider common variants like 'user_session'
        cand = COLUMN_GUESS['session'] + ['user_session', 'sessionID']
        cfg.session_col = pick(cand)
    if cfg.user_col is None:
        cfg.user_col = pick(COLUMN_GUESS['user'])
    if cfg.event_type_col is None:
        cfg.event_type_col = pick(COLUMN_GUESS['event'] + ['eventType'])
    if cfg.product_col is None:
        cfg.product_col = pick(COLUMN_GUESS['product'] + ['product'])
    if cfg.category_col is None:
        cfg.category_col = pick(['category_id', 'category', 'cat_id', 'cid'])
    if cfg.time_col is None:
        cfg.time_col = pick(COLUMN_GUESS['time'] + ['eventTime'])
    if cfg.price_col is None:
        cfg.price_col = pick(COLUMN_GUESS['price'])

    missing = []
    for k, v in [('session', cfg.session_col), ('event', cfg.event_type_col), ('product', cfg.product_col), ('time', cfg.time_col)]:
        if v is None:
            missing.append(k)
    if missing:
        raise ValueError(f"Missing required columns not found automatically: {missing}. Please set them via args.")
    return cfg


def build_vocab(train_col: pd.Series, test_col: pd.Series) -> Dict:
    vals = pd.concat([train_col, test_col], axis=0).fillna('NA').astype(str).unique().tolist()
    vocab = {v: i+1 for i, v in enumerate(vals)}  # 0 reserved for PAD
    return vocab


def sessions_to_sequences(df: pd.DataFrame, cfg: Config,
                          event_vocab: Dict, product_vocab: Dict,
                          max_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays of shape [num_sessions, max_len]
    - event_ids, product_ids, time_deltas (scaled), price_seq (optional -> zeros if missing)
    """
    df = df.sort_values([cfg.session_col, cfg.time_col])

    # Parse as datetime with UTC, then drop tz to get naive epoch seconds
    t_dt = pd.to_datetime(df[cfg.time_col], errors='coerce', utc=True)

    if t_dt.notna().any():
    # convert to naive UTC and take epoch seconds
        t_naive = t_dt.dt.tz_convert('UTC').dt.tz_localize(None)
        df['_t'] = t_naive.view('int64') // 10**9
    else:
    # try numeric timestamps
        t_num = pd.to_numeric(df[cfg.time_col], errors='coerce')
        if t_num.isna().all():
        # fallback: just use row order
            t_num = pd.Series(np.arange(len(df)), index=df.index, dtype='int64')
        df['_t'] = t_num.values

    grp = df.groupby(cfg.session_col)
    sessions = list(grp.indices.keys())

    E = np.zeros((len(sessions), max_len), dtype=np.int32)
    P = np.zeros((len(sessions), max_len), dtype=np.int32)
    DT = np.zeros((len(sessions), max_len), dtype=np.float32)
    PR = np.zeros((len(sessions), max_len), dtype=np.float32)

    for i, sid in enumerate(sessions):
        g = grp.get_group(sid)
        e = g[cfg.event_type_col].fillna('NA').astype(str).map(lambda x: event_vocab.get(x, 0)).values
        p = g[cfg.product_col].fillna('NA').astype(str).map(lambda x: product_vocab.get(x, 0)).values
        tt = g['_t'].values
        dt = np.diff(np.r_[tt[0], tt])  # deltas
        if cfg.price_col is not None and cfg.price_col in g.columns:
            pr = pd.to_numeric(g[cfg.price_col], errors='coerce').fillna(0.0).values
        else:
            pr = np.zeros_like(e, dtype=np.float32)

        # keep last max_len events
        if len(e) > max_len:
            e = e[-max_len:]
            p = p[-max_len:]
            dt = dt[-max_len:]
            pr = pr[-max_len:]
        # pad left
        L = len(e)
        E[i, -L:] = e
        P[i, -L:] = p
        if dt.max() > 0:
            DT[i, -L:] = dt / (dt.std() + 1e-6)
        else:
            DT[i, -L:] = 0
        PR[i, -L:] = pr if L > 0 else 0

    return sessions, E, P, DT, PR


def make_session_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Session-level engineered features to help both branches and the blender."""
    df = df.copy()
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors='coerce')

    # Basic counts
    agg = {
        cfg.event_type_col: ['count', pd.Series.nunique],
        cfg.product_col: [pd.Series.nunique],
    }
    if cfg.category_col is not None and cfg.category_col in df.columns:
        agg[cfg.category_col] = [pd.Series.nunique]
    if cfg.price_col is not None and cfg.price_col in df.columns:
        agg[cfg.price_col] = ['sum', 'mean', 'max']

    g = df.groupby(cfg.session_col).agg(agg)
    base_cols = ['event_count', 'event_type_nunique', 'product_nunique']
    if cfg.category_col is not None and cfg.category_col in df.columns:
        base_cols.append('category_nunique')
    price_cols = (['price_sum', 'price_mean', 'price_max'] if (cfg.price_col and cfg.price_col in df.columns) else [])
    g.columns = base_cols + price_cols

    # duration
    tmin = df.groupby(cfg.session_col)[cfg.time_col].min()
    tmax = df.groupby(cfg.session_col)[cfg.time_col].max()
    dur = (tmax - tmin).dt.total_seconds().fillna(0).to_frame('duration_sec')

    feats = g.join(dur)

    # rates & flags
    if cfg.event_type_col in df.columns:
        ev_counts = df.pivot_table(index=cfg.session_col, columns=cfg.event_type_col, values=cfg.product_col, aggfunc='count', fill_value=0)
        ev_counts.columns = [f"evcnt_{str(c)}" for c in ev_counts.columns]
        feats = feats.join(ev_counts, how='left')
        feats['view_rate'] = feats.filter(like='evcnt_view').sum(axis=1) / (feats['event_count'] + 1e-6)
        feats['cart_rate'] = feats.filter(like='evcnt_cart').sum(axis=1) / (feats['event_count'] + 1e-6)
        feats['purchase_rate'] = feats.filter(like='evcnt_purchase').sum(axis=1) / (feats['event_count'] + 1e-6)
        feats['has_purchase'] = (feats.filter(like='evcnt_purchase').sum(axis=1) > 0).astype(int)
        feats['has_cart'] = (feats.filter(like='evcnt_cart').sum(axis=1) > 0).astype(int)

    feats['unique_product_ratio'] = feats['product_nunique'] / (feats['event_count'] + 1e-6)
    if 'category_nunique' in feats.columns:
        feats['unique_category_ratio'] = feats['category_nunique'] / (feats['event_count'] + 1e-6)

    feats = feats.fillna(0)
    return feats


# ----------------------------- Extra Feature: Bigrams ----------------------------- #

def build_top_bigrams(df: pd.DataFrame, cfg: Config, topk: int = 12) -> List[str]:
    """Find top-K event bigrams on train to add as per-session counts."""
    tmp = df[[cfg.session_col, cfg.event_type_col, cfg.time_col]].copy()
    tmp = tmp.sort_values([cfg.session_col, cfg.time_col])
    bigrams = {}
    for sid, g in tmp.groupby(cfg.session_col):
        ev = g[cfg.event_type_col].astype(str).tolist()
        for a, b in zip(ev[:-1], ev[1:]):
            key = f"bg_{a}__{b}"
            bigrams[key] = bigrams.get(key, 0) + 1
    top = sorted(bigrams.items(), key=lambda x: -x[1])[:topk]
    return [k for k, _ in top]


def add_bigram_features(df: pd.DataFrame, cfg: Config, top_bigrams: List[str]) -> pd.DataFrame:
    tmp = df[[cfg.session_col, cfg.event_type_col, cfg.time_col]].copy()
    tmp = tmp.sort_values([cfg.session_col, cfg.time_col])
    rows = []
    for sid, g in tmp.groupby(cfg.session_col):
        ev = g[cfg.event_type_col].astype(str).tolist()
        counts = {k: 0 for k in top_bigrams}
        for a, b in zip(ev[:-1], ev[1:]):
            key = f"bg_{a}__{b}"
            if key in counts:
                counts[key] += 1
        counts[cfg.session_col] = sid
        rows.append(counts)
    out = pd.DataFrame(rows).set_index(cfg.session_col)
    out = out.reindex(pd.Index(df[cfg.session_col].unique())).fillna(0)
    return out

# ----------------------------- PyTorch Model ------------------------------ #

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:, :L]


class SessionTransformer(nn.Module):
    def __init__(self, n_event: int, n_product: int, cfg: Config, use_price: bool = True):
        super().__init__()
        self.cfg = cfg
        #Embedding / projections
        self.event_emb = nn.Embedding(n_event + 1, cfg.d_model // 3, padding_idx=0)
        self.product_emb = nn.Embedding(n_product + 1, cfg.d_model // 3, padding_idx=0)
        self.time_proj = nn.Linear(1, cfg.d_model // 6)
        self.price_proj = nn.Linear(1, cfg.d_model // 6) if use_price else None

        # >>> GİRİŞ BOYUTUNU OTOMATİK HESAPLA
        self.d_in = (
            self.event_emb.embedding_dim
            + self.product_emb.embedding_dim
            + self.time_proj.out_features
            + (self.price_proj.out_features if self.price_proj is not None else 0))

        self.proj = nn.Linear(self.d_in, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, max_len=cfg.max_len + 5)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        self.attn_pool = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model//2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model//2, 1)
        )

    def forward(self, E, P, DT, PR, mask):
        # E,P: [B,L] int; DT,PR: [B,L] float; mask: [B,L] bool (True for keep)
        e = self.event_emb(E)
        p = self.product_emb(P)
        dt = self.time_proj(DT.unsqueeze(-1))
        pieces = [e, p, dt]
        if self.price_proj is not None:
            pieces.append(self.price_proj(PR.unsqueeze(-1)))
        x = torch.cat(pieces, dim=-1)
        x = self.proj(x)
        x = self.pos(x)

        key_padding_mask = ~mask  # True where pad
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # attention pooling
        att = self.attn_pool(x).squeeze(-1)  # [B,L]
        att = att.masked_fill(~mask, -1e9)
        w = torch.softmax(att, dim=1).unsqueeze(-1)  # [B,L,1]
        pooled = (x * w).sum(dim=1)
        out = self.head(pooled).squeeze(-1)
        return out


class SessionDataset(Dataset):
    def __init__(self, E, P, DT, PR, mask, y=None):
        self.E = E; self.P = P; self.DT = DT; self.PR = PR; self.mask = mask; self.y = y
    def __len__(self):
        return self.E.shape[0]
    def __getitem__(self, idx):
        out = {
            'E': torch.tensor(self.E[idx], dtype=torch.long),
            'P': torch.tensor(self.P[idx], dtype=torch.long),
            'DT': torch.tensor(self.DT[idx], dtype=torch.float32),
            'PR': torch.tensor(self.PR[idx], dtype=torch.float32),
            'mask': torch.tensor(self.mask[idx], dtype=torch.bool)
        }
        if self.y is not None:
            out['y'] = torch.tensor(self.y[idx], dtype=torch.float32)
        return out


# ----------------------------- Training loop ----------------------------- #

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    def forward(self, pred, target):
        err = pred - target
        abs_err = torch.abs(err)
        quad = torch.minimum(abs_err, torch.tensor(self.delta, device=pred.device))
        lin = abs_err - quad
        loss = 0.5 * quad**2 + self.delta * lin
        return loss.mean()


def get_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_seed_fold(cfg: Config, seed: int, fold: int,
                        E_tr, P_tr, DT_tr, PR_tr, mask_tr, y_tr,
                        E_va, P_va, DT_va, PR_va, mask_va, y_va,
                        n_event: int, n_product: int,
                        use_price: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    seed_everything(seed)

    train_ds = SessionDataset(E_tr, P_tr, DT_tr, PR_tr, mask_tr, y_tr)
    valid_ds = SessionDataset(E_va, P_va, DT_va, PR_va, mask_va, y_va)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SessionTransformer(n_event, n_product, cfg, use_price=use_price).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_pct)
    sch = get_scheduler(opt, total_steps, warmup_steps)

    crit = HuberLoss(cfg.huber_delta)

    best_rmse = 1e18
    best_state = None
    patience, best_epoch, es_pat = 8, -1, 0

    for epoch in range(cfg.epochs):
        model.train()
        tr_losses = []
        for batch in train_loader:
            E_b = batch['E'].to(cfg.device)
            P_b = batch['P'].to(cfg.device)
            DT_b = batch['DT'].to(cfg.device)
            PR_b = batch['PR'].to(cfg.device)
            M_b = batch['mask'].to(cfg.device)
            y_b = batch['y'].to(cfg.device)

            opt.zero_grad(set_to_none=True)
            pred = model(E_b, P_b, DT_b, PR_b, M_b)
            loss = crit(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sch.step()
            tr_losses.append(loss.item())

        # validate
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in valid_loader:
                E_b = batch['E'].to(cfg.device)
                P_b = batch['P'].to(cfg.device)
                DT_b = batch['DT'].to(cfg.device)
                PR_b = batch['PR'].to(cfg.device)
                M_b = batch['mask'].to(cfg.device)
                y_b = batch['y'].to(cfg.device)
                p = model(E_b, P_b, DT_b, PR_b, M_b)
                val_preds.append(p.detach().cpu().numpy().reshape(-1))

            vpred = np.concatenate(val_preds)
            y_va_np = y_va.reshape(-1)
            rmse = np.sqrt(mean_squared_error(y_va_np, vpred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            es_pat = 0
        else:
            es_pat += 1
        print(f"Seed {seed} Fold {fold} Epoch {epoch+1}/{cfg.epochs} | Train {np.mean(tr_losses):.4f} | Val RMSE {rmse:.4f} | Best {best_rmse:.4f} @e{best_epoch+1}")
        if es_pat >= patience:
            print("Early stopping.")
            break

    # load best
    model.load_state_dict(best_state)

    # predict on valid & return model for test use
    model.eval()
    with torch.no_grad():
        # valid preds
        valid_pred = []
        for batch in valid_loader:
            E_b = batch['E'].to(cfg.device)
            P_b = batch['P'].to(cfg.device)
            DT_b = batch['DT'].to(cfg.device)
            PR_b = batch['PR'].to(cfg.device)
            M_b = batch['mask'].to(cfg.device)
            p = model(E_b, P_b, DT_b, PR_b, M_b)
            valid_pred.append(p.detach().cpu().numpy())
        valid_pred = np.concatenate(valid_pred)

    return valid_pred, best_state


# --------------------------- CatBoost (optional) -------------------------- #

def catboost_branch(train_df_sess: pd.DataFrame, test_df_sess: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from catboost import CatBoostRegressor, Pool
    except Exception as e:
        print("CatBoost not available, skipping tree branch.", e)
        return None, None

    X = train_df_sess.drop(columns=[cfg.target_col])
    y = train_df_sess[cfg.target_col].astype(float).values
    X_test = test_df_sess.copy()

    # infer categoricals by dtype or name
    cat_cols = [c for c in X.columns if X[c].dtype == 'object' or c.endswith('_cat')]

    kf = KFold(n_splits=cfg.folds, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))

    params = dict(
        depth=8,
        learning_rate=0.05,
        iterations=50000,
        loss_function='RMSE',
        eval_metric='RMSE',
        l2_leaf_reg=4.0,
        random_seed=42,
        od_type='Iter',
        od_wait=800,
        verbose=False
    )

    for fold, (tr, va) in enumerate(kf.split(X), 1):
        tr_pool = Pool(X.iloc[tr], y[tr], cat_features=cat_cols)
        va_pool = Pool(X.iloc[va], y[va], cat_features=cat_cols)
        model = CatBoostRegressor(**params)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        oof[va] = model.predict(va_pool)
        pred += model.predict(Pool(X_test, cat_features=cat_cols)) / cfg.folds
        rmse = np.sqrt(mean_squared_error(y[va], oof[va]))
        print(f"CatBoost Fold {fold} RMSE: {rmse:.5f}")

    rmse_oof = np.sqrt(mean_squared_error(y, oof))
    print(f"CatBoost OOF RMSE: {rmse_oof:.5f}")
    return oof, pred


# ------------------------------- Main flow -------------------------------- #

def main(args):
    cfg = Config(
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        warmup_pct=args.warmup_pct,
        grad_clip=args.grad_clip,
        folds=args.folds,
        use_log1p=not args.no_log1p,
        blend_weight_transformer=args.blend,
        # CLI-provided column overrides
        session_col=args.session_col,
        user_col=args.user_col,
        event_type_col=args.event_col,
        product_col=args.product_col,
        category_col=args.category_col,
        time_col=args.time_col,
        price_col=args.price_col,
    )

    id_col, target_col = infer_id_target('sample_submission.csv')
    cfg.id_col = id_col
    cfg.target_col = target_col

    print(f"ID col: {cfg.id_col} | Target col: {cfg.target_col}")

    train_df = pd.read_csv('train.csv')
    test_df  = pd.read_csv('test.csv')

    # auto-detect required columns
    cfg = auto_detect_columns(train_df, cfg)
    print("Detected columns:")
    print({
        'session': cfg.session_col,
        'user': cfg.user_col,
        'event': cfg.event_type_col,
        'product': cfg.product_col,
        'category': cfg.category_col,
        'time': cfg.time_col,
        'price': cfg.price_col
    })

    # Build vocabs on train+test for events/products
    event_vocab = build_vocab(train_df[cfg.event_type_col], test_df[cfg.event_type_col])
    product_vocab = build_vocab(train_df[cfg.product_col], test_df[cfg.product_col])
    n_event = len(event_vocab) + 1
    n_product = len(product_vocab) + 1

    # Build sequences
    tr_sessions, E_tr, P_tr, DT_tr, PR_tr = sessions_to_sequences(train_df, cfg, event_vocab, product_vocab, cfg.max_len)
    te_sessions, E_te, P_te, DT_te, PR_te = sessions_to_sequences(test_df,  cfg, event_vocab, product_vocab, cfg.max_len)

    # masks (non-zero event or product implies true)
    mask_tr = (E_tr != 0) | (P_tr != 0)
    mask_te = (E_te != 0) | (P_te != 0)

    # session-level features (for CatBoost and potential blender)
    sess_feats_tr = make_session_features(train_df, cfg)
    sess_feats_te = make_session_features(test_df, cfg)

    # Optional: add bigram + last-event features for CatBoost branch
    if getattr(args, 'advfeats', False):
        top_bgs = build_top_bigrams(train_df, cfg, topk=getattr(args, 'topk_bigrams', 12))
        print(f"Top bigrams: {top_bgs}")
        bg_tr = add_bigram_features(train_df, cfg, top_bgs)
        bg_te = add_bigram_features(test_df, cfg, top_bgs)
        last_ev_tr = train_df.sort_values([cfg.session_col, cfg.time_col]).groupby(cfg.session_col)[cfg.event_type_col].last().astype(str)
        last_ev_te = test_df.sort_values([cfg.session_col, cfg.time_col]).groupby(cfg.session_col)[cfg.event_type_col].last().astype(str)
        ltr = pd.get_dummies(last_ev_tr, prefix='last_ev')
        lte = pd.get_dummies(last_ev_te, prefix='last_ev')
        ltr, lte = ltr.align(lte, join='outer', axis=1, fill_value=0)
        sess_feats_tr = sess_feats_tr.join([bg_tr, ltr], how='left').fillna(0)
        sess_feats_te = sess_feats_te.join([bg_te, lte], how='left').fillna(0)

    # target vector aligned to session order in sequence arrays
    # train_df should have target per session; if target is row-level, we need a session aggregation.
    if cfg.target_col in train_df.columns:
        # if target is row-level duplicated per row, take last or max; otherwise expect a separate target table
        tgt_series = train_df.groupby(cfg.session_col)[cfg.target_col].last()
        y_raw = tgt_series.loc[tr_sessions].values.astype(float)
    else:
        raise ValueError(f"Target column {cfg.target_col} not found in train.csv.")

    # log1p target
    if cfg.use_log1p:
        y = np.log1p(y_raw)
    else:
        y = y_raw.copy()

    # prepare CV groups if available
    if cfg.user_col is not None and cfg.user_col in train_df.columns:
        groups = train_df.groupby(cfg.session_col)[cfg.user_col].first().loc[tr_sessions].values
        splitter = GroupKFold(n_splits=cfg.folds)
        split_iter = splitter.split(tr_sessions, y, groups=groups)
    else:
        splitter = KFold(n_splits=cfg.folds, shuffle=True, random_state=42)
        split_iter = splitter.split(tr_sessions, y)

    # map session index to row index in sess_feats
    tr_indexer = pd.Index(tr_sessions)
    te_indexer = pd.Index(te_sessions)

    # OOF containers
    oof_transformer = np.zeros(len(tr_sessions))
    test_pred_transformer = np.zeros(len(te_sessions))

    # Train transformer with seeds & folds
    for seed in cfg.seeds:
        seed_everything(seed)
        for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
            # We must regenerate split_iter for each seed
            if isinstance(splitter, GroupKFold):
                split_iter = splitter.split(tr_sessions, y, groups=groups)
                for i, (a, b) in enumerate(split_iter, 1):
                    if i == fold:
                        tr_idx, va_idx = a, b
                        break
            else:
                kf = KFold(n_splits=cfg.folds, shuffle=True, random_state=seed)
                for i, (a, b) in enumerate(kf.split(tr_sessions, y), 1):
                    if i == fold:
                        tr_idx, va_idx = a, b
                        break

            vpred, best_state = train_one_seed_fold(
                cfg, seed, fold,
                E_tr[tr_idx], P_tr[tr_idx], DT_tr[tr_idx], PR_tr[tr_idx], mask_tr[tr_idx], y[tr_idx],
                E_tr[va_idx], P_tr[va_idx], DT_tr[va_idx], PR_tr[va_idx], mask_tr[va_idx], y[va_idx],
                n_event, n_product,
                use_price=(cfg.price_col is not None)
            )
            oof_transformer[va_idx] += vpred / len(cfg.seeds)

            # Build test loader for this best model
            model = SessionTransformer(n_event, n_product, cfg, use_price=(cfg.price_col is not None))
            model.load_state_dict(best_state)
            model.to(cfg.device)
            model.eval()
            ds_te = SessionDataset(E_te, P_te, DT_te, PR_te, mask_te)
            dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            preds_te = []
            with torch.no_grad():
                for batch in dl_te:
                    E_b = batch['E'].to(cfg.device)
                    P_b = batch['P'].to(cfg.device)
                    DT_b = batch['DT'].to(cfg.device)
                    PR_b = batch['PR'].to(cfg.device)
                    M_b = batch['mask'].to(cfg.device)
                    p = model(E_b, P_b, DT_b, PR_b, M_b)
                    preds_te.append(p.detach().cpu().numpy())
            test_pred_transformer += np.concatenate(preds_te) / (cfg.folds * len(cfg.seeds))

        # reset splitter for next seed
        if cfg.user_col is not None and cfg.user_col in train_df.columns:
            split_iter = splitter.split(tr_sessions, y, groups=groups)
        else:
            split_iter = splitter.split(tr_sessions, y)

    # Inverse transform log target for OOF
    oof_t = np.expm1(oof_transformer) if cfg.use_log1p else oof_transformer
    rmse_oof_t = np.sqrt(mean_squared_error(y_raw, oof_t))
    print(f"Transformer OOF RMSE (real scale): {rmse_oof_t:.5f}")


    # CatBoost on session features (if available)
    train_sess = sess_feats_tr.copy()
    train_sess[cfg.target_col] = y_raw
    test_sess = sess_feats_te.copy()

    oof_tree, pred_tree = catboost_branch(train_sess, test_sess, cfg)

    # Blending
    if (oof_tree is not None) and (pred_tree is not None):
        # Align oof_tree to same order as sessions
        # train_sess index is session_id; ensure same order
        oof_tree_aligned_vals = pd.Series(oof_tree, index=train_sess.index).loc[tr_sessions].values

        # Blend oof for score check
        blend_w = cfg.blend_weight_transformer
        oof_blend = blend_w * oof_t + (1.0 - blend_w) * oof_tree_aligned_vals
        rmse_blend = np.sqrt(mean_squared_error(y_raw, oof_blend))
        print(f"Blended OOF RMSE (w={blend_w:.2f}): {rmse_blend:.5f}")

        # Blend test preds
        test_tree_aligned = pd.Series(pred_tree, index=test_sess.index).loc[te_sessions].values
        # test_sess order is te_sessions order
        test_pred = (
        blend_w * (np.expm1(test_pred_transformer) if cfg.use_log1p else test_pred_transformer)
        + (1.0 - blend_w) * test_tree_aligned
        )
    else:
        print("CatBoost kolu pas geçildi (kurulu değil ya da başarısız). Transformer tahminleri kullanılacak.")
        test_pred = np.expm1(test_pred_transformer) if cfg.use_log1p else test_pred_transformer

    # Write submission
    sample = pd.read_csv('sample_submission.csv')
    sub = sample.copy()

    # map predictions to the order of sample IDs; assume sample has id_col that equals session ids
    pred_map = pd.Series(test_pred, index=te_sessions)
    sub[cfg.target_col] = pred_map.loc[sub[cfg.id_col]].values

    out_name = 'submission_podium.csv'
    sub.to_csv(out_name, index=False)
    print(f"Saved {out_name} with {len(sub)} rows.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Column overrides (optional)
    parser.add_argument('--session_col', type=str, default=None, help='session id column (e.g., user_session)')
    parser.add_argument('--event_col', type=str, default=None, help='event type column (e.g., event_type)')
    parser.add_argument('--product_col', type=str, default=None, help='product id column (e.g., product_id)')
    parser.add_argument('--category_col', type=str, default=None, help='category id column (e.g., category_id)')
    parser.add_argument('--time_col', type=str, default=None, help='event time column (e.g., event_time)')
    parser.add_argument('--user_col', type=str, default=None, help='user id column')
    parser.add_argument('--price_col', type=str, default=None, help='price column (optional)')

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--warmup_pct', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--folds', type=int, default=7)
    parser.add_argument('--no_log1p', action='store_true', help='disable log1p target')
    parser.add_argument('--blend', type=float, default=0.70, help='initial weight for Transformer in blend')
    parser.add_argument('--huber_delta', type=float, default=1.5)
    parser.add_argument('--seeds', type=str, default='42,77,123,2025')
    parser.add_argument('--advfeats', action='store_true', help='enable extra bigram + last-event features for CatBoost branch')
    parser.add_argument('--topk_bigrams', type=int, default=12)

    args = parser.parse_args()


    main(args)
