"""Spectral embedding via the symmetric normalized Laplacian.

Computes the bottom-k eigenvectors of
    L_sym = I - D^{-1/2} A D^{-1/2}
where A is the (weighted) adjacency matrix and D is the diagonal degree
matrix. The eigenvectors corresponding to the smallest eigenvalues are
stacked as columns of an n-by-k embedding matrix, which can then be fed
to a geometric clusterer such as k-means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

AdjacencyLike = Union[np.ndarray, sp.spmatrix]


def _to_csr(A: AdjacencyLike) -> sp.csr_matrix:
    """Coerce an adjacency-like input to a symmetric CSR matrix."""
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    else:
        A = A.tocsr()
    # Symmetrize defensively; user input may be slightly non-symmetric due to
    # construction order, but L_sym is only well-defined on undirected graphs.
    if (A != A.T).nnz != 0:
        A = ((A + A.T) * 0.5).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def normalized_laplacian(A: AdjacencyLike) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build L_sym = I - D^{-1/2} A D^{-1/2} as a sparse matrix.

    Isolated nodes (degree 0) get D^{-1/2} = 0 by convention, which makes
    their corresponding row/column of L_sym equal to e_i and gives them an
    eigenvalue of 1 — they form their own trivial component.
    """
    A = _to_csr(A)
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L_sym = sp.identity(n, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt
    # Re-symmetrize to kill floating-point drift from the triple product.
    L_sym = ((L_sym + L_sym.T) * 0.5).tocsr()
    return L_sym, deg


@dataclass
class SpectralEmbedding:
    """Result of a bottom-k eigendecomposition of L_sym."""

    eigenvalues: np.ndarray  # shape (k,), ascending
    embedding: np.ndarray    # shape (n, k), columns are eigenvectors
    degrees: np.ndarray      # shape (n,), original node degrees


def spectral_embedding(
    A: AdjacencyLike,
    k: int,
    *,
    row_normalize: bool = False,
    tol: float = 0.0,
    max_iter: int | None = None,
    seed: int | None = None,
) -> SpectralEmbedding:
    """Compute the bottom-k eigenvectors of the symmetric normalized Laplacian.

    Parameters
    ----------
    A : array or sparse matrix, shape (n, n)
        Symmetric adjacency matrix of the graph (weighted or unweighted).
    k : int
        Number of eigenvectors to return. Must satisfy 1 <= k < n.
    row_normalize : bool, default False
        If True, normalize each row of the embedding to unit L2 norm
        (Ng-Jordan-Weiss). Useful when feeding the embedding into k-means.
    tol : float, default 0.0
        Convergence tolerance for ARPACK (0 means machine precision).
    max_iter : int or None, default None
        ARPACK iteration cap. None lets ARPACK pick its default (10*n).
    seed : int or None
        Seed for the ARPACK starting vector, for reproducibility.

    Returns
    -------
    SpectralEmbedding with eigenvalues sorted ascending and the matching
    eigenvectors stacked column-wise.
    """
    L_sym, deg = normalized_laplacian(A)
    n = L_sym.shape[0]
    if not (1 <= k < n):
        raise ValueError(f"k must satisfy 1 <= k < n; got k={k}, n={n}")

    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n)

    # L_sym is PSD with eigenvalues in [0, 2]. eigsh's shift-invert mode
    # (sigma=0) is the most reliable way to grab the bottom of the spectrum,
    # but it requires solving linear systems with L_sym. For graphs with
    # isolated components L_sym is exactly singular at sigma=0, so we fall
    # back to the spectral-shift trick: largest eigvals of (2I - L_sym)
    # correspond to smallest eigvals of L_sym, and ARPACK is happy on the
    # top end without a factorization.
    M = 2.0 * sp.identity(n, format="csr") - L_sym
    vals_shifted, vecs = eigsh(
        M,
        k=k,
        which="LA",
        tol=tol,
        maxiter=max_iter,
        v0=v0,
    )
    eigvals = 2.0 - vals_shifted
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    vecs = vecs[:, order]
    # Tiny negative values from floating-point error — clip to 0.
    eigvals = np.clip(eigvals, 0.0, None)

    if row_normalize:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

    return SpectralEmbedding(eigenvalues=eigvals, embedding=vecs, degrees=deg)


if __name__ == "__main__":
    # Quick sanity check: a graph of two disjoint cliques should produce two
    # near-zero eigenvalues and an embedding that separates the cliques.
    n_per = 20
    block = np.ones((n_per, n_per)) - np.eye(n_per)
    A = np.block([[block, np.zeros_like(block)],
                  [np.zeros_like(block), block]])
    # Add a single bridging edge so the graph is connected.
    A[0, n_per] = A[n_per, 0] = 1.0

    res = spectral_embedding(A, k=3, row_normalize=False, seed=0)
    print("Bottom-3 eigenvalues:", np.round(res.eigenvalues, 6))
    print("Fiedler vector sign split (cluster sizes):",
          int(np.sum(res.embedding[:, 1] > 0)),
          int(np.sum(res.embedding[:, 1] <= 0)))
