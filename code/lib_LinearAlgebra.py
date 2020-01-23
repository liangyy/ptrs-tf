import tensorflow as tf
import numpy as np

class DataScheme:
    '''
    X is predictor matrix
    Y is outcome / covariate matrix
    '''
    def __init__(self, dataset = None, X_index = None, Y_index = None, outcome_indice = None, covariate_indice = None):
        self.dataset = dataset
        self.X_index = None
        self.Y_index = None
        self.outcome_indice = None
        self.covariate_indice = None
        self._update_index(X_index, 'X_index')
        self._update_index(Y_index, 'Y_index')
        self._update_indice(outcome_indice, 'outcome_indice')
        self._update_indice(covariate_indice, 'covariate_indice')
    def get_num_predictor(self):
        if self.X_index is None:
            return None
        else:
            return self.dataset.element_spec[self.X_index].shape[-1]
    def get_num_outcome(self):
        if self.outcome_indice is None:
            return None
        else:
            return len(self.outcome_indice)
    def get_num_covariate(self):
        if self.covariate_indice is None:
            return None
        else:
            return len(self.covariate_indice)
    def _update_index(self, index, object_name):
        if self.dataset is None:
            setattr(self, object_name, None)
        elif len(self.dataset.element_spec) > index:
            setattr(self, object_name, index)
        else:
            raise ValueError(f'{object_name} is bigger than the length of dataset')
    def _update_indice(self, indice, object_name):
        if self.Y_index is None:
            setattr(self, object_name, None)
        else:
            n_y = self.dataset.element_spec[self.Y_index].shape[-1]
            if min(indice) < 0:
                raise ValueError(f'{object_name} cannot be smaller than 0')
            elif max(indice) >= n_y:
                raise ValueError(f'{object_name} cannot exceed than length of Y')
            else:
                setattr(self, object_name, indice)


class SVDInstance:
    def __init__(self, rcond = 1e-10):
        self.rcond = 1e-10
        self.s = None
        self.v = None
        self.d = None
    def solve(self, mat):
        s, self.u, self.v = tf.linalg.svd(mat)
        # Ignore singular values close to zero to prevent numerical overflow
        limit = self.rcond * tf.reduce_max(mat)
        non_zero = tf.greater(s, limit)
        self.s = tf.where(non_zero, s, tf.zeros(s.shape))
            
class LeastSquaredEstimator:
    def __init__(self, data_scheme, rcond = 1e-10, intercept = False):
        self.rcond = rcond
        self.intercept = intercept
        self.data_scheme = data_scheme
        self.xtx = None
        self.xty = None
        self.svd = SVDInstance(self.rcond)
        self.betahat = None
    def _out_dim(self):
        if self.data_scheme is None:
            return None
        else:
            n_predictor = self.data_scheme.get_num_predictor()
            n_covariate = self.data_scheme.get_num_covariate()
            n_outcome = self.data_scheme.get_num_outcome()
            x_dim = n_predictor + n_covariate
            y_dim = n_outcome
            if self.intercept is True:
                x_dim += 1  
            return (x_dim, y_dim)
    def _init_xtx_xty(self):
        xdim, ydim = self._out_dim()
        self.xtx = tf.Variable(initial_value = tf.zeros([xdim, xdim], tf.float32))
        self.xty = tf.Variable(initial_value = tf.zeros([xdim, ydim], tf.float32))
    def solve(self):
        if self.data_scheme is None:
            raise ValueError('data_scheme is None, we cannot solve')
        self._init_xtx_xty() 
        for ele in self.data_scheme.dataset:
            x = ele[self.data_scheme.X_index]
            y = ele[self.data_scheme.Y_index]
            covar = tf.gather(y, self.data_scheme.covariate_indice, axis = 1)
            y = tf.gather(y, self.data_scheme.outcome_indice, axis = 1) 
            x = tf.concat((x, covar), axis = 1)
            if self.intercept is True:
                x_with_1 = tf.concat((tf.ones([x.shape[0], 1], tf.float32), x), axis = 1)
                self.xtx.assign(
                    tf.add(self.xtx, tf.matmul(tf.transpose(x_with_1), x_with_1))
                )
                self.xty.assign(
                    tf.add(self.xty, tf.matmul(tf.transpose(x_with_1), y))
                )
            elif self.intercept is False:
                self.xtx.assign(tf.add(self.xtx, tf.matmul(tf.transpose(x), x)))
                self.xty.assign(tf.add(self.xty, tf.matmul(tf.transpose(x), y)))
        
        # svd on xtx
        self.svd.solve(self.xtx)
        
        # calculate beta hat
        self.betahat = tf.matmul(
            tf.matmul(
                tf.matmul(
                    self.svd.v, 
                    tf.linalg.tensor_diag(tf.math.reciprocal_no_nan(self.svd.s))
                ), 
                tf.transpose(self.svd.u)
            )
            , self.xty
        )

    
