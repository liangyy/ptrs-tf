import tensorflow as tf

'''
-----------------------
  Implementation Note
-----------------------
This library implement Elastic Net solver for [X, y1, ..., yP]
    Y = M + X B + E, E ~ normal (independent between rows and column-specific variance)
Or equivalently
    yp = mp + X bp + ep
with hyperparameters lambda1, ..., lambdaP and the shared alpha
    l1 = lambda * alpha
    l2 = lambda * (1 - alpha)
and objective is
    obj = 1 / n || yp - mp - X bp ||_2^2 + l1 ||bp||_1 + l2 ||bp||_2^2 

Utility functions:
0. prod(U, V) = colMean(U * V)
1. g(bp)    = 1 / n || yp - mp - X bp ||_2^2 + l2 ||bp||_2^2  
   vec_g(B) = 1 / n 
            ( prod(B, X^t X B) 
            - 2 prod(B, X^t Y_) 
            + prod(Y_, Y_) ) 
            + prod(B, B) L2 (with Y_ = Y - M) 
2. vec_grad_g(B) = 2 / n ( X^t X B - X^t Y_ ) + 2 * B L2 
3. prox(U, K) = SoftThresholdByColumn(U, K)
4. update(B, T, lambda) = prox(B - T vec_grad_g(B), T * lambda) 

Update rule (coordinate descent over M and B):
1. M^+ = colMean(Y) - colMean(X) B
2. Y_ = Y - M^+
3. B^+ = prox_lambda_T(B - vec_grad_g(B) T) with line search

Line search given:
0.1. gg = vec_g(B)
0.2. ggrad = vec_grad_g(B)
1.   T = T0
2.   B(T) = update(B, T, lambda)
3.   TGT = B - B(T)
4.   LHS = vec_g(B(T))
5.   RHS = gg - ggrad * TGT + TGT * TGT / 2 / T
6.   shrink T if necessary
'''

class ElasticNetPlus:
    def __init__(self, num_predictors, num_outcomes, alpha, Lambda_, B = None, M = None, seed = None):
        '''
        B for betahat matrix
        M for intercept matrix (it is actually n x 1)
        Lambda_ is numpy array with shape = (num_outcomes, ) 
        '''
        init = tf.random_normal_initializer(seed = seed)
        self.num_predictors = num_predictors
        self.num_outcomes = num_outcomes
        self.__init_param('B', B, init, num_predictors, num_outcomes)
        self.__init_param('M', M, init, 1, num_outcomes)
        self.alpha = alpha
        self.__init_lambda(Lambda_)
        self.update_l1_l2()   
    def update_l1_l2(self):
        l1 = self.alpha * self.Lambda_
        l2 = (1 - self.alpha) * self.Lambda_
        self.l1 = tf.constant(tf.cast(l1, tf.float32))
        self.l2 = tf.constant(tf.cast(l2, tf.float32))
    def update_lambda(self, Lambda_):
        self._init_lambda(Lambda_)
        self.update_l1_l2()
    def one_step_update(self, x, y):
        # TODO
    def __init_param(self, var_name, var_value, init, nrow, ncol):
        if var_value is None:
            var_value = tf.Variable(init(shape = [nrow, ncol]), name = var_name)
        setattr(self, var_name, var_value)
    def __init_lambda(self, Lambda_):
        if Lambda_.shape[0] != self.num_outcomes or len(Lambda_.shape) > 1:
            ValueError('Lambda_ is illegal. We requires numpy array with shape = (num_outcomes, )')
        self.Lambda_ = Lambda_ 
    