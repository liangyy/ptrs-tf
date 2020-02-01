import scipy.stats
import numpy as np
def inv_norm_col(mat):
    return np.apply_along_axis(inv_norm_vec, 0, mat)
def inv_norm_vec(vec, offset = 1):
    rank = myrank(vec)
    return scipy.stats.norm.ppf(rank / (len(rank) + offset), loc = 0, scale = 1)
def myrank(vec):
    argsort = np.argsort(vec)
    ranks = np.empty_like(argsort)
    ranks[argsort] = np.arange(len(vec))
    return ranks + 1  # rank starts from 1
def quick_partial_r2(x, y, yhat):
    '''
    x: 
        i = sample id
        j = covariate id
    y: 
        i = sample id
        k = outcome id
    yhat:
        i = sample id
        k = outcome id
        p = predictor id
    return partial r2 of yhat_p on y accounting for x
    return shape: k x p
    formula:
        Hhat = x_ (x_^t x_)^{-1} x_^t
        where x_ = [1, x]
        t0 = y^t y
        k0 = y^t Hhat y
        t1 = yhat_p^t yhat_p
        k1 = yhat_p^t Hhat yhat_p
        t01 = y^t yhat_p
        k01 = y^t Hhat yhat_p
        partial_r2 = (t01 - k01)^2  / (t0 - k0) / (t1 - k1)
    optimized procedure:
        xty = x_^t y
        xtyhat = x_^t yhat
        xtx_inv = inv(xtx)
        k0 = xty xtx_inv xty
        k1 = xtyhat xtx_inv xtyhat
        k01 = xty solve(xtx, xtyhat)
        as intermediate step
    '''
    i, j, k, p = _quick_partial_r2_check_dim(x, y, yhat)
    x_ = np.concatenate(
        (
            np.ones(shape = (i, 1)),
            x
        ),
        axis = 1
    )
    xtx = np.einsum('ji,jk', x_, x_)
    # xtx_inv_xt = np.linalg.solve(xtx, x_.transpose())
    # x_xtx_inv_xt = np.einsum('ij,jk', x_, xtx_inv_xt)
    xty = np.einsum('ji,jk', x_, y)
    xtyhat = np.einsum('ji,jkp', x_, yhat)
    xtx_inv = np.linalg.inv(xtx)
    xtx_inv_xty = np.einsum('ij,jk', xtx_inv, xty)
    xtx_inv_xtyhat = np.einsum('ij,jkp', xtx_inv, xtyhat)
    # compute k's
    k0 = np.einsum('ji,ji->i', xty, xtx_inv_xty)
    k1 = np.einsum('jip,jip->ip', xtyhat, xtx_inv_xtyhat)
    k01 = np.einsum('ji,jip->ip', xty, xtx_inv_xtyhat)
    # compute t's
    t0 = np.einsum('ij,ij->j', y, y)
    t1 = np.einsum('ijp,ijp->jp', yhat, yhat)
    t01 = np.einsum('ij,ijp->jp', y, yhat)
    # calculate partial r2
    partial_r2_nomi = np.power(t01 - k01, 2)
    partial_r2_deno0 = t0 - k0
    partial_r2_deno1 = t1 - k1
    partial_r2_deno = np.einsum('i,ij->ij', partial_r2_deno0, partial_r2_deno1)
    partial_r2 = partial_r2_nomi / partial_r2_deno
    return partial_r2
