import tensorflow as tf
import numpy as np
import h5py 
import re

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
        self.num_predictors = self.get_num_predictor()
    def get_data_matrix(self, element, only_x = False):
        x = element[self.X_index]
        y = element[self.Y_index]
        if only_x is False:
            covar = tf.gather(y, self.covariate_indice, axis = 1)
            x = tf.concat((x, covar), axis = 1)
        y = tf.gather(y, self.outcome_indice, axis = 1) 
        return x, y
    def get_indice_x(self):
        '''
        return indice in data matrix that are for x
        '''
        n_x = self.num_predictors
        start = 0
        end = n_x
        return [ i for i in range(start, end) ]
    def get_indice_covar(self):
        '''
        return indice in data matrix that are for covariates
        '''
        n_x = self.num_predictors
        n_covar = self.get_num_covariate()
        start = n_x
        end = n_x + n_covar
        return [ i for i in range(start, end) ]
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
            n_predictor = self.data_scheme.num_predictors
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
    def _prep_for_intercept(self, x):
        if self.intercept is True:
            x = tf.concat((tf.ones([x.shape[0], 1], tf.float32), x), axis = 1)
        return x
    def _reshape_y(self, y_list):
        y_list = np.array(y_list)
        return np.reshape(y_list, [y_list.shape[0] * y_list.shape[1], y_list.shape[2]], order = 'C')
    def _get_betahat_by_indice(self, indices):
        if self.intercept is True:
            indices = [ i + 1 for i in indices ]
        return tf.gather(self.betahat, indices, axis = 0) 
    def get_betahat_x(self):
        indices = self.data_scheme.get_indice_x()
        return self._get_betahat_by_indice(indices)
    def get_betahat_covar(self):
        indices = self.data_scheme.get_indice_covar()
        return self._get_betahat_by_indice(indices)
    def get_intercept(self):
        if self.intercept is True:
            intercept = self.betahat[0, :]
        else:
            intercept = None
        return intercept
    def solve(self):
        if self.data_scheme is None:
            raise ValueError('data_scheme is None, we cannot solve')
        self._init_xtx_xty()
        n_processed = 0
        for ele in self.data_scheme.dataset:
            x, y = self.data_scheme.get_data_matrix(ele)
            x = self._prep_for_intercept(x)
            n_new = x.shape[0]
            n_old = n_processed
            # val_old_mean * (n_old / (n_old + n_new)) + val_new_sum / (n_old + n_new) 
            f_old = n_old / (n_old + n_new)
            f_new = 1 / (n_old + n_new)
            self.xtx.assign(tf.add(
                    tf.multiply(self.xtx, f_old), 
                    tf.multiply(tf.matmul(tf.transpose(x), x), f_new)
                )
            )
            self.xty.assign(tf.add(
                    tf.multiply(self.xty, f_old), 
                    tf.multiply(tf.matmul(tf.transpose(x), y), f_new)
                )
            )
            n_processed += n_new
        
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
    def predict(self, dataset):
        '''
        We assume dataset has the same data_scheme as self.data_scheme.
        y_pred = ((intercept), x, covar) %*% betahat
        It returns y, ypred as numpy array
        '''
        if self.betahat is None:
            raise ValueError('betahat is None. We cannot predict')
        y_ = []
        y_pred_ = []
        for ele in dataset:
            x, y = self.data_scheme.get_data_matrix(ele)
            x = self._prep_for_intercept(x)
            y_pred_.append(tf.matmul(x, self.betahat))
            y_.append(y)
        y_pred_ = self._reshape_y(y_pred_)
        y_ = self._reshape_y(y_)
        return {'y_pred': y_pred_, 'y': y_}
    def predict_x(self, dataset):
        '''
        We assume dataset has the same data_scheme as self.data_scheme.
        y_pred = x %*% betahat_x
        It returns y, ypred as numpy array
        '''
        if self.betahat is None:
            raise ValueError('betahat is None. We cannot predict')
        y_ = []
        y_pred_ = []
        for ele in dataset:
            x, y = self.data_scheme.get_data_matrix(ele, only_x = True)
            y_pred_.append(tf.matmul(x, self.get_betahat_x()))
            y_.append(y)
        y_pred_ = self._reshape_y(y_pred_)
        y_ = self._reshape_y(y_)
        return {'y_pred_from_x': y_pred_, 'y': y_}    
    def minimal_save(self, filename, save_inner_product = False):
        '''
        Perform minimal save, which saves the minimal things needed for prediction. 
        They are: 
        1) xtx, xty (Optional, controlled by save_inner_product. 
        These are computationally intensive and could be large.); 
        2) betahat; 
        3) all members but dataset in data_scheme;
        4) intercept
        '''
        save_dic = {}
        if save_inner_product is True:
            save_dic['xtx'] = self.xtx.numpy()
            save_dic['xty'] = self.xty.numpy()
        save_dic['betahat'] = self.betahat.numpy()
        save_dic['intercept'] = self.intercept * 1
        for i in self.data_scheme.__dict__.keys():
            if i != 'dataset':
                save_dic['data_scheme.' + i] = getattr(self.data_scheme, i)
            else:
                save_dic['data_scheme.' + i] = b'save_mode'
        with h5py.File(filename, 'w') as f:
            for i in save_dic.keys():
                print('Saving {}'.format(i))
                f.create_dataset(i, data = save_dic[i])
    def minimal_load(self, filename):
        '''
        Load HDF5 generated by `minimal_save`.
        Note that it is not a perfect load and it is minimal in the sense that it allows `predict` and `predict_x` to work properly.
        '''
        with h5py.File(filename, 'r') as f:
            data_scheme = DataScheme()
            for i in f.keys():
                if 'data_scheme.' in i:
                    mem = re.sub('data_scheme.', '', i)
                    if mem == 'outcome_indice' or mem == 'covariate_indice':
                        val = list(f[i][...])
                    elif mem == 'X_index' or mem == 'Y_index' or mem == 'num_predictors':
                        val = f[i][...]
                    setattr(data_scheme, mem, val)
                else:
                    if i == 'intercept':
                        if f[i][...] == 1:
                            setattr(self, i, True)
                        elif f[i][...] == 0:
                            setattr(self, i, False)
                    elif i == 'betahat':
                        setattr(self, i, tf.constant(f[i][:], tf.float32))
        self.data_scheme = data_scheme
                      
                
            
    
