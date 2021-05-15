# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 11:48:42 2020

@author: hhu1
"""
#import warnings
from math import sqrt
import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs

def _cholesky_omp(X, y, n_nonzero_coefs, tol=None, copy_X=True,
                  return_path=False):
    """Orthogonal Matching Pursuit step using the Cholesky decomposition.
    X : dictionary
    y : sample
    n_nonzero_coefs: Targeted number of non-zero elements
    tol : Targeted squared error, if not None overrides n_nonzero_coefs.
    Returns
    -------
    gamma: Non-zero elements of the solution
    idx :  Indice in the solution vector
    coef : array, shape (n_features, n_nonzero_coefs)
        The first k values of column k correspond to the coefficient value
        for the active features at that step. The lower left triangle contains
        garbage. Only returned if ``return_path=True``.
    n_active : int
        Number of sparse coefficients at convergence.
    """
    if copy_X:
        X = X.copy('F')
    else:  # even if we are allowed to overwrite, still copy it if bad order
        X = np.asfortranarray(X)

    min_float = np.finfo(X.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(('nrm2', 'swap'), (X,))
    potrs, = get_lapack_funcs(('potrs',), (X,))

    alpha = np.dot(X.T, y) # X: 100*84 y: one sample 100*1 alpha^zero
    residual = y #100*1
    gamma = np.empty(0) #8*1
    n_active = 0 # Number of active features at convergence.
    indices = np.arange(X.shape[1])  # keeping track of swapping 84*1

    max_features = X.shape[1] if tol is not None else n_nonzero_coefs # number of sparse coefficient

    L = np.empty((max_features, max_features), dtype=X.dtype) #8*8

    if return_path:
        coefs = np.empty_like(L)

    while True:
        lam = np.argmax(np.abs(np.dot(X.T, residual))) # lam control the index of selected atom
        if lam < n_active or alpha[lam] ** 2 < min_float:
            # atom already selected or inner product too small
           # warnings.warn(premature, RuntimeWarning, stacklevel=2)
            break

        if n_active > 0:
            # Updates the Cholesky decomposition of X' X
            L[n_active, :n_active] = np.dot(X[:, :n_active].T, X[:, lam])
            linalg.solve_triangular(L[:n_active, :n_active],
                                    L[n_active, :n_active],
                                    trans=0, lower=1,
                                    overwrite_b=True,
                                    check_finite=False)
            v = nrm2(L[n_active, :n_active]) ** 2 #2-norm
            Lkk = linalg.norm(X[:, lam]) ** 2 - v
            if Lkk <= min_float:  # selected atoms are dependent
                #warnings.warn(premature, RuntimeWarning, stacklevel=2)
                print("Value is too small")
                break
            L[n_active, n_active] = sqrt(Lkk)
        else:
            L[0, 0] = linalg.norm(X[:, lam])

        X.T[n_active], X.T[lam] = swap(X.T[n_active], X.T[lam])
        alpha[n_active], alpha[lam] = alpha[lam], alpha[n_active]
        indices[n_active], indices[lam] = indices[lam], indices[n_active]
        n_active += 1

        # solves LL'x = X'y as a composition of two triangular systems
        gamma, _ = potrs(L[:n_active, :n_active], alpha[:n_active], lower=True,
                         overwrite_b=False)

        if return_path:
            coefs[:n_active, n_active - 1] = gamma
        residual = y - np.dot(X[:, :n_active], gamma)
        if tol is not None and nrm2(residual) ** 2 <= tol:
            break
        elif n_active == max_features:
            break

    if return_path:
        return gamma, indices[:n_active], coefs[:, :n_active], n_active
    else:
        return gamma, indices[:n_active], n_active
    
#-------------------------------Orthogonal Matching Pursuit (OMP)---------------------------------------------
def orthogonal_mp(X, y, n_nonzero_coefs, tol=None, precompute=False,
                  copy_X=True, return_path=False,
                  return_n_iter=False):
    """
    X : dictionary
    y : sample
    n_nonzero_coefs : Desired number of non-zero entries in the solution. 
    tol : Maximum norm of the residual. If not None, overrides n_nonzero_coefs.
    copy_X : bool, optional
    
    Returns
    -------
    coef : array, shape (n_features,) or (n_features, n_targets)
        Coefficients of the OMP solution. If `return_path=True`, this contains
        the whole coefficient path. In this case its shape is
        (n_features, n_features) or (n_features, n_targets, n_features) and
        iterating over the last axis yields coefficients in increasing order
        of active features.

    n_iters : array-like or int
        Number of active features across every target. Returned only if
        `return_n_iter` is set to True.
    """
#    X = check_array(X, order='F', copy=copy_X) #(100*84)
    copy_X = False
    if y.ndim == 1: #(100,50)
        y = y.reshape(-1, 1)
#   y = check_array(y)
    if y.shape[1] > 1:  # subsequent targets will be affected
        copy_X = True
    if n_nonzero_coefs is None and tol is None:
        print("n_nonzero_coefs is none")
        # default for n_nonzero_coefs is 0.1 * n_features
        # but at least one.
        #n_nonzero_coefs =max(int(0.1 * X.shape[1]), 1)/2 #DECIDE the sparsity
    if tol is not None and tol < 0:
        raise ValueError("Epsilon cannot be negative")
    if tol is None and n_nonzero_coefs <= 0:
        raise ValueError("The number of atoms must be positive")
    if tol is None and n_nonzero_coefs > X.shape[1]:
        raise ValueError("The number of atoms cannot be more than the number "
                         "of features")
    if return_path:
        coef = np.zeros((X.shape[1], y.shape[1], X.shape[1]))
    else:
        coef = np.zeros((X.shape[1], y.shape[1]))
    n_iters = []

    for k in range(y.shape[1]):
        out = _cholesky_omp(  #construct the sparse coefficients
            X, y[:, k], n_nonzero_coefs, tol,
            copy_X=copy_X, return_path=return_path)
        if return_path:
            _, idx, coefs, n_iter = out
            coef = coef[:, :, :len(idx)]
            for n_active, x in enumerate(coefs.T):
                coef[idx[:n_active + 1], k, n_active] = x[:n_active + 1]
        else:
            x, idx, n_iter = out
            coef[idx, k] = x
        n_iters.append(n_iter)

    if y.shape[1] == 1:
        n_iters = n_iters[0]
    if return_n_iter:
        return np.squeeze(coef), n_iters
    else:
        return np.squeeze(coef)