import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _NeuMFNet(nn.Module):
    def __init__(self, n_users, n_items, dim, mlp_layers, dropout):
        super().__init__()
        self.u_gmf, self.i_gmf = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
        self.u_mlp, self.i_mlp = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
        layers, d = [], 2 * dim
        for h in mlp_layers:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dim + d, 1)
        for e in (self.u_gmf, self.i_gmf, self.u_mlp, self.i_mlp):
            nn.init.normal_(e.weight, std=0.01)

    def forward(self, u, i):
        g = self.u_gmf(u) * self.i_gmf(i)
        m = self.mlp(torch.cat([self.u_mlp(u), self.i_mlp(i)], -1))
        return self.out(torch.cat([g, m], -1)).squeeze(-1)


class NeuMF:

    def __init__(self, n_users, n_items, dim=64, mlp_layers=(128, 64, 32), dropout=0.1, n_neg=4,
                 epochs=20, batch=4096, lr=1e-3, wd=0.0, seed=0, device="cuda"):
        self.__dict__.update(locals()); del self.__dict__["self"]

    def fit(self, df):
        _seed(self.seed)
        dev = self.device
        self.net = _NeuMFNet(self.n_users, self.n_items, self.dim, self.mlp_layers, self.dropout).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        u_pos = torch.as_tensor(df.userId.to_numpy(), device=dev)
        i_pos = torch.as_tensor(df.movieId.to_numpy(), device=dev)
        seen = torch.zeros(self.n_users, self.n_items, dtype=torch.bool, device=dev)
        seen[u_pos, i_pos] = True
        n = len(u_pos)
        self.net.train()
        for _ in range(self.epochs):
            u_neg = u_pos.repeat(self.n_neg)
            i_neg = torch.randint(0, self.n_items, (len(u_neg),), device=dev)
            bad = seen[u_neg, i_neg]
            while bad.any():
                i_neg[bad] = torch.randint(0, self.n_items, (int(bad.sum()),), device=dev)
                bad = seen[u_neg, i_neg]
            u = torch.cat([u_pos, u_neg]); i = torch.cat([i_pos, i_neg])
            y = torch.cat([torch.ones(n, device=dev), torch.zeros(len(u_neg), device=dev)])
            perm = torch.randperm(len(u), device=dev)
            for s in range(0, len(u), self.batch):
                b = perm[s:s + self.batch]
                loss = F.binary_cross_entropy_with_logits(self.net(u[b], i[b]), y[b])
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def score(self, users):
        self.net.eval()
        items = torch.arange(self.n_items, device=self.device)
        out = []
        for u in users:
            uu = torch.full((self.n_items,), int(u), device=self.device)
            out.append(self.net(uu, items).float().cpu().numpy())
        return np.stack(out)


class _SASRecNet(nn.Module):
    def __init__(self, n_items, max_len, dim, n_blocks, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(dim, n_heads, dim * 4, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_blocks)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("causal", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1))

    def forward(self, seq):
        L = seq.size(1)
        h = self.item(seq) * self.item.embedding_dim ** 0.5 + self.pos(torch.arange(L, device=seq.device))
        m = self.causal[:L, :L] | (seq == 0)[:, None, :]
        m = m & ~torch.eye(L, dtype=torch.bool, device=seq.device)
        h = self.enc(self.drop(h), mask=m.repeat_interleave(self.n_heads, 0))
        return self.norm(h)

    def logits(self, h):
        return h @ self.item.weight[1:].T


class SASRec:

    def __init__(self, n_users, n_items, max_len=200, dim=64, n_blocks=2, n_heads=2, dropout=0.2,
                 epochs=200, batch=128, lr=1e-3, wd=0.0, seed=0, device="cuda"):
        self.__dict__.update(locals()); del self.__dict__["self"]

    def _seqs(self, df):
        d = df.sort_values(["userId", "time_idx"])
        return {u: g.to_numpy() + 1 for u, g in d.groupby("userId").movieId}

    def _pad(self, s):
        s = s[-self.max_len:]
        return np.pad(s, (self.max_len - len(s), 0))

    def fit(self, df):
        _seed(self.seed)
        dev = self.device
        self.hist = self._seqs(df)
        self.net = _SASRecNet(self.n_items, self.max_len, self.dim, self.n_blocks, self.n_heads, self.dropout).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        L1, windows = self.max_len + 1, []
        for s in self.hist.values():
            for end in range(len(s), 1, -self.max_len):
                w = s[max(0, end - L1):end]
                windows.append(np.pad(w, (L1 - len(w), 0)))
        W = torch.as_tensor(np.stack(windows), device=dev)
        self.net.train()
        for _ in range(self.epochs):
            for b in torch.randperm(len(W), device=dev).split(self.batch):
                x, y = W[b, :-1], W[b, 1:]
                h = self.net(x)
                m = (y > 0) & (x > 0)
                loss = F.cross_entropy(self.net.logits(h[m]), y[m] - 1)
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def score(self, users):
        self.net.eval()
        seq = torch.as_tensor(np.stack([self._pad(self.hist.get(u, np.zeros(0, dtype=np.int64))) for u in users]),
                              device=self.device)
        out = [self.net.logits(self.net(seq[s:s + 256])[:, -1]).cpu().numpy() for s in range(0, len(seq), 256)]
        return np.concatenate(out)
