import numpy as np
import scipy.sparse as sp


def interaction_matrix(df, n_users, n_items, min_rating=None):
    d = df if min_rating is None else df[df.rating >= min_rating]
    return sp.csr_matrix(
        (np.ones(len(d), dtype=np.float32), (d.userId.to_numpy(), d.movieId.to_numpy())),
        shape=(n_users, n_items),
    )


class Popularity:
    def __init__(self, n_users, n_items):
        self.n_users, self.n_items = n_users, n_items

    def fit(self, df):
        self.pop = np.bincount(df.movieId, minlength=self.n_items).astype(np.float32)
        return self

    def score(self, users):
        return np.tile(self.pop, (len(users), 1))


class EASE:

    def __init__(self, n_users, n_items, lam=500.0, min_rating=None, tau=None):
        self.n_users, self.n_items, self.lam, self.min_rating, self.tau = n_users, n_items, lam, min_rating, tau

    def fit(self, df):
        self.X = interaction_matrix(df, self.n_users, self.n_items, self.min_rating)
        if self.tau is not None:
            age = df.groupby("userId").time_idx.transform("max") - df.time_idx
            w = np.exp(-age.to_numpy() / self.tau).astype(np.float32)
            self.profile = sp.csr_matrix((w, (df.userId.to_numpy(), df.movieId.to_numpy())),
                                         shape=(self.n_users, self.n_items))
        G = (self.X.T @ self.X).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        return self

    def score(self, users):
        X = self.X if self.tau is None else self.profile
        return np.asarray(X[users] @ self.B)
