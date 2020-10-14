import scipy.stats
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
import statsmodels.api as sm

def inv_norm_col(mat, exclude_idx = None):
    if exclude_idx is None:
        return np.apply_along_axis(inv_norm_vec, 0, mat)
    else:
        out = np.apply_along_axis(inv_norm_vec, 0, mat)
        out[:, exclude_idx] = mat[:, exclude_idx]
        return out

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
    print(xtx.shape)
    xtx_inv = np.linalg.pinv(xtx)
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
def _quick_partial_r2_check_dim(x, y, yp):
    if len(x.shape) != 2:
        raise ValueError('Wrong x shape')
    if len(y.shape) != 2:
        raise ValueError('Wrong y shape')
    if len(yp.shape) != 3:
        raise ValueError('Wrong yp shape')
    ix, jx = x.shape
    iy, ky = y.shape
    iyp, kyp, pyp = yp.shape
    if ix != iy or ix != iyp:
        raise ValueError('Wrong dim-1 in x, y, yp do not match')
    if ky != kyp:
        raise ValueError('Wrong dim-2 in y, yp do not match')
    return ix, jx, ky, pyp

def binary_perf(x, y, yhat, func=calc_partial_r2_logistic):
    '''
    x: covariate
    y: y observed 
    yhat: y predicted
    Fit:
    * model0: yobs ~ 1 + covariate
    * model1: yobs ~ 1 + covariate + ypred
    * partial_R2: 1 - lld1 / lld0
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
    '''
    i, j, k, p = _quick_partial_r2_check_dim(x, y, yhat)
    # add intercept
    x_ = np.concatenate(
        (np.ones(shape = (i, 1)), x),
        axis = 1
    )
    y_round = round_y_to_binary(y)
    partial_r2 = np.zeros((k, p))
    for ki in range(k):
        for pi in range(p):
            yp = yhat[:, ki, pi]
            yo = y[:, ki]
            partial_r2[ki, pi] = func(yp, yo, x_)
    return partial_r2

def round_y_to_binary(y):
    yround = np.round_(y)
    n_not_binary = np.logical_not(np.logical_or(yround == 0, yround == 1))
    if n_not_binary != 0:
        raise ValueError("We need y to be binary 0/1. But there are {} not following the rule.".format())
    return yround

def get_logistic_lld(y, x):
    mod = sm.Logit(y, x)
    res = mod.fit(method='newton')
    yp = res.predict(x)
    return log_loss(y, yp)
    
def calc_partial_r2_logistic(yp, yo, covar):
    # log_loss returns -log likelihood / nsample
    lld0 = get_logistic_lld(yo, covar)
    tmp = np.concatenate((covar, yp), axis=1)
    lld1 = get_logistic_lld(yo, tmp)
    return 1 - lld1 / lld0

def calc_auc(yp, yo):
    return roc_auc_score(yp, yo)        
    