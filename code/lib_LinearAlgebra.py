import tensorflow as tf
import numpy as np
import h5py 
import re
import functools
import lib_ElasticNet, util_ElasticNet

class DataScheme:
    '''
    X is predictor matrix
    Y is outcome / covariate matrix
    '''
    def __init__(self, dataset = None, X_index = None, Y_index = None, outcome_indice = None, covariate_indice = None, x_indice = None):
        self.dataset = dataset
        self.X_index = None
        self.Y_index = None
        self.outcome_indice = None
        self.covariate_indice = None
        self.x_indice = None
        self._update_index(X_index, 'X_index')
        self._update_index(Y_index, 'Y_index')
        self._update_indice(outcome_indice, 'outcome_indice')
        self._update_indice(covariate_indice, 'covariate_indice')
        self._update_indice(x_indice, 'x_indice', type = 'X')
        # self.num_predictors = self.get_num_predictor()
    def get_data_matrix(self, element, only_x = False, only_covar = False):
        y = element[self.Y_index]
        if only_covar is False:
            x = element[self.X_index]
            if self.x_indice is not None:
                x = tf.gather(x, self.x_indice, axis = 1)
            if only_x is False:
                covar = tf.gather(y, self.covariate_indice, axis = 1)
                x = tf.concat((x, covar), axis = 1)
        else:
            x = tf.gather(y, self.covariate_indice, axis = 1)
        y = tf.gather(y, self.outcome_indice, axis = 1) 
        return x, y
    def get_data_matrix_x_in_cnn(self, element):
        x = element[self.X_index]
        if self.x_indice is not None:
            x = tf.gather(x, self.x_indice, axis = 1)
        y = element[self.Y_index]
        covar = tf.gather(y, self.covariate_indice, axis = 1)
        y = tf.gather(y, self.outcome_indice, axis = 1) 
        return [tf.expand_dims(x, axis = 2), covar], y
    def get_indice_x(self):
        '''
        return indice in data matrix that are for x
        '''
        n_x = self.get_num_predictor()  # self.num_predictors
        start = 0
        end = n_x
        return [ i for i in range(start, end) ]
    def get_indice_covar(self):
        '''
        return indice in data matrix that are for covariates
        '''
        n_x = self.get_num_predictor()  # self.num_predictors
        n_covar = self.get_num_covariate()
        start = n_x
        end = n_x + n_covar
        return [ i for i in range(start, end) ]
    def get_num_predictor(self):
        if self.X_index is None:
            return None
        else:
            if self.x_indice is None:
                return self.dataset.element_spec[self.X_index].shape[-1]
            else:
                return len(self.x_indice)
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
    def _update_indice(self, indice, object_name, type = 'Y'):
        if self.Y_index is None:
            setattr(self, object_name, None)
        else:
            if type == 'Y':
                n_y = self.dataset.element_spec[self.Y_index].shape[-1]
                if min(indice) < 0:
                    raise ValueError(f'{object_name} cannot be smaller than 0')
                elif max(indice) >= n_y:
                    raise ValueError(f'{object_name} cannot exceed than length of Y')
                else:
                    setattr(self, object_name, indice)
            elif type == 'X':
                n_x = self.dataset.element_spec[self.X_index].shape[-1]
                if indice is None:
                    setattr(self, object_name, indice)
                    return 
                if min(indice) < 0:
                    raise ValueError(f'{object_name} cannot be smaller than 0')
                elif max(indice) >= n_x:
                    raise ValueError(f'{object_name} cannot exceed than length of X')
                else:
                    setattr(self, object_name, indice)


class SVDInstance:
    def __init__(self, rcond = 1e-10):
        self.rcond = 1e-10
        self.s = None
        self.v = None
        self.d = None
    def solve(self, mat):
        s, u, v = tf.linalg.svd(mat)
        mat_rank = tf.linalg.matrix_rank(mat).numpy()
        # Thresholding by rank
        # discard_idx = tf.where(range(len(s)) < mat_rank)[:, 0]
        s = s[:mat_rank]
        u = u[:, : mat_rank]
        v = v[:, : mat_rank]
        # Ignore singular values close to zero to prevent numerical overflow
        limit = self.rcond * tf.reduce_max(mat)
        non_zero = tf.greater(s, limit)
        self.s = tf.where(non_zero, s, tf.zeros(s.shape))
        self.u = u
        self.v = v

class BatchNormalizer:
    def __init__(self, scheme_func, dataset, shuffle = 100):
        self.shuffle = shuffle
        self.mean, self.std = self._init_mean_and_std(scheme_func, dataset)
    def _init_mean_and_std(self, scheme_func, dataset):
        batch_dataset = dataset.take(100).shuffle(100).take(1)
        for ele in batch_dataset:
            x, _ = scheme_func(ele)
            batch_mean = tf.reduce_mean(x, axis = 0)
            break
        for ele in batch_dataset:
            x, _ = scheme_func(ele)
            batch_sq_error = tf.math.squared_difference(x, batch_mean)
            break
        batch_std = tf.math.sqrt(tf.reduce_mean(batch_sq_error, axis = 0))
        return batch_mean, batch_std
    def apply(self, x):
        return tf.math.divide_no_nan(tf.math.subtract(x, self.mean), self.std)
        
class FullNormalizer:
    def __init__(self, scheme_func, dataset, tensor = False):
        self.mean, self.std = self._init_mean_and_std(scheme_func, dataset, tensor = tensor)
    def _init_mean_and_std(self, scheme_func, dataset, tensor = False):
        if tensor is False:
            mean = 0
            n_processed = 0
            for ele in dataset:
                x, _ = scheme_func(ele)
                # tf.print(type(x) is list)
                if type(x) is list:
                    x = tf.concat((x[0][:, :, 0], x[1]), axis = 1)
                n_old = 0
                n_this = x.shape[0]
                f_old = n_old / (n_old + n_this)
                # mean_new = mean_old * n_old / (n_old + n_this) + mean_this * n_this / (n_old + n_this)
                mean = mean * f_old + tf.reduce_mean(x, axis = 0) * (1 - f_old)
                n_processed += n_this
            mse = 0
            n_processed = 0
            for ele in dataset:
                x, _ = scheme_func(ele)
                if type(x) is list:
                    x = tf.concat((x[0][:, :, 0], x[1]), axis = 1)
                n_old = 0
                n_this = x.shape[0]
                f_old = n_old / (n_old + n_this)
                # mse_new = mse_old * n_old / (n_old + n_this) + mse_this * n_this / (n_old + n_this)
                mse_this = tf.reduce_mean(
                    tf.math.squared_difference(x, mean), 
                    axis = 0
                )
                mse = mse * f_old + mse_this * (1 - f_old)
            std = tf.math.sqrt(mse)
            return mean, std
        else:
            x, _ = scheme_func(dataset)
            if type(x) is list:
                x = tf.concat((x[0][:, :, 0], x[1]), axis = 1)
            mean = tf.reduce_mean(x, axis = 0)
            mse = tf.reduce_mean(
                tf.math.squared_difference(x, mean), 
                axis = 0
            )
            std = tf.math.sqrt(mse)
            return mean, std
    def apply(self, x):
        if type(x) is list:
            x[0] = x[0][:, :, 0]
            s0 = x[0].shape[1]
            o0 = tf.math.divide_no_nan(tf.math.subtract(x[0], self.mean[:s0]), self.std[:s0])
            o1 = tf.math.divide_no_nan(tf.math.subtract(x[1], self.mean[s0:]), self.std[s0:])
            return [tf.expand_dims(o0, axis = 2), o1]
        else:
            return tf.math.divide_no_nan(tf.math.subtract(x, self.mean), self.std)
        
class _nested_y_DataScheme:
    def __init__(self, dataset = None, X_index = None, Y_index = None, predictor_indice = None, outcome_indice = None, covariate_indice = None):
        self.dataset = dataset
        self.X_index = X_index
        self.Y_index = Y_index
        self.predictor_indice = predictor_indice
        self.outcome_indice = outcome_indice
        self.covariate_indice = covariate_indice
        self.num_predictors = self.get_num_predictor()
    def update_predictor_indice(self, predictor_indice):
        self.predictor_indice = predictor_indice
        self.num_predictors = self.get_num_predictor()
    def get_data_matrix(self, element):
        y = element[self.Y_index[0]][self.Y_index[1]]
        covar = tf.gather(y, self.covariate_indice, axis = 1)
        y = tf.gather(y, self.outcome_indice, axis = 1) 
        if self.predictor_indice is None:
            x = covar
        else:
            x = element[self.X_index]
            x = tf.gather(x, self.predictor_indice, axis = 1) 
            x = tf.concat((x, covar), axis = 1)
        return x, y
    def _get_indice_length(self, indice):
        if indice is None:
            return 0
        else:
            return len(indice)
    def get_num_outcome(self):
            return self._get_indice_length(self.outcome_indice)
    def get_num_covariate(self):
            return self._get_indice_length(self.covariate_indice)
    def get_num_predictor(self):
            return self._get_indice_length(self.predictor_indice)
            
class LeastSquaredEstimator:
    def __init__(self, data_scheme, normalizer = False, rcond = 1e-10, intercept = False):
        self.normalizer = normalizer
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
    def _prep_for_intercept(self, x):
        if self.intercept is True:
            x = tf.concat((tf.ones([x.shape[0], 1], tf.float32), x), axis = 1)
        return x
    def _reshape_y(self, y_list):
        return np.concatenate(y_list, axis = 0)
        # y_list = np.array(y_list)
        # return np.reshape(y_list, [y_list.shape[0] * y_list.shape[1], y_list.shape[2]], order = 'C')
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
    def solve(self, logging = None, sample_size = None, scaling = True, return_normalizer = False):
        if self.data_scheme is None:
            raise ValueError('data_scheme is None, we cannot solve')
        if logging is not None and sample_size is not None:
            timer = 0
        if self.normalizer == True:
            normalizer = FullNormalizer(self.data_scheme.get_data_matrix, self.data_scheme.dataset)
            # print(normalizer.mean)
            # print(normalizer.std)
        self._init_xtx_xty()
        n_processed = 0
        for ele in self.data_scheme.dataset:
            x, y = self.data_scheme.get_data_matrix(ele)
            if self.normalizer == True:
                x = normalizer.apply(x)
            x = self._prep_for_intercept(x)
            n_new = x.shape[0]
            n_old = n_processed
            # val_old_mean * (n_old / (n_old + n_new)) + val_new_sum / (n_old + n_new) 
            if scaling is True:
                f_old = n_old / (n_old + n_new)
                f_new = 1 / (n_old + n_new)
            else:
                f_old = 1
                f_new = 1
            self.xtx.assign(tf.add(
                    tf.multiply(self.xtx, f_old), 
                    tf.matmul(tf.multiply(tf.transpose(x), f_new), x)
                )
            )
            self.xty.assign(tf.add(
                    tf.multiply(self.xty, f_old), 
                    tf.matmul(tf.multiply(tf.transpose(x), f_new), y)
                )
            )
            n_processed += n_new
            if logging is not None and sample_size is not None:
                percent_5_fold = int(n_processed / sample_size / 0.05)
                percent_ = percent_5_fold * 5
                if percent_5_fold > timer:
                    logging.info(f'Progress {percent_}%: {n_processed} / {sample_size}') 
                    timer = percent_5_fold
        
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

        if return_normalizer == True:
            return normalizer
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
        if self.normalizer == True:
            normalizer = FullNormalizer(self.data_scheme.get_data_matrix, dataset)
        for ele in dataset:
            x, y = self.data_scheme.get_data_matrix(ele)
            if self.normalizer == True:
                x = normalizer.apply(x)
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
        if self.normalizer == True:
            scheme_func = functools.partial(self.data_scheme.get_data_matrix, only_x = True)
            normalizer = FullNormalizer(scheme_func, dataset)
            # print(normalizer.mean)
            # print(normalizer.std)              
        for ele in dataset:
            x, y = self.data_scheme.get_data_matrix(ele, only_x = True)
            if self.normalizer == True:
                x = normalizer.apply(x)
            y_pred_.append(tf.matmul(x, self.get_betahat_x()))
            y_.append(y)
        y_pred_ = self._reshape_y(y_pred_)
        y_ = self._reshape_y(y_)
        return {'y_pred_from_x': y_pred_, 'y': y_} 
    def partial_r2(self, dataset, batch_size = 128, logging = None):
        pred = self.predict_x(dataset)
        ## setup for new data scheme
        yy_pred = tf.data.Dataset.from_tensor_slices(pred['y_pred_from_x'])
        new_data = tf.data.Dataset.zip((dataset.unbatch(), yy_pred)).batch(batch_size)
        X_index = 1
        Y_index = (0, 1)
        covariate_indice = self.data_scheme.covariate_indice
        scheme_i = _nested_y_DataScheme(
            dataset = new_data, 
            X_index = X_index,
            Y_index = Y_index,
            covariate_indice = covariate_indice
        )
        ## END
        n_total = pred['y_pred_from_x'].shape[1]
        result = np.empty((n_total, 3))
        result[:] = np.nan
        for pred_i in range(n_total):
            if logging is not None:
                logging.info(f'Partial R2 Processing {pred_i} / {n_total}')
                logging.info('now processing outcome index {}'.format(self.data_scheme.outcome_indice[pred_i]))
            scheme_i.outcome_indice = [self.data_scheme.outcome_indice[pred_i]]
            scheme_i.update_predictor_indice([pred_i])
            solve_full = LeastSquaredEstimator(scheme_i, intercept = True, normalizer = True)
            solve_full.solve()
            out_full = solve_full.predict(scheme_i.dataset)
            sse_full = tf.reduce_sum(tf.math.squared_difference(out_full['y'], out_full['y_pred']))
            scheme_i.update_predictor_indice(None)
            solve_null = LeastSquaredEstimator(scheme_i, intercept = True, normalizer = True)
            solve_null.solve()
            out_null = solve_null.predict(scheme_i.dataset)
            sse_null = tf.reduce_sum(tf.math.squared_difference(out_null['y'], out_null['y_pred']))
            R2 = 1 - sse_full / sse_null
            result[pred_i, :] = [R2, sse_full, sse_null]
        return result             
    def minimal_save(self, filename, save_inner_product = False):
        '''
        Perform minimal save, which saves the minimal things needed for prediction. 
        They are: 
        1) xtx, xty (Optional, controlled by save_inner_product. 
        These are computationally intensive and could be large.); 
        2) betahat; 
        3) all members but dataset in data_scheme;
        4) intercept
        5) batch normalization shuffle
        '''
        save_dic = {}
        if save_inner_product is True:
            save_dic['xtx'] = self.xtx.numpy()
            save_dic['xty'] = self.xty.numpy()
        save_dic['betahat'] = self.betahat.numpy()
        save_dic['intercept'] = self.intercept * 1
        save_dic['normalizer'] = self.normalizer * 1
        for i in self.data_scheme.__dict__.keys():
            if i != 'dataset':
                save_dic['data_scheme.' + i] = getattr(self.data_scheme, i)
            else:
                save_dic['data_scheme.' + i] = b'save_mode'
        with h5py.File(filename, 'w') as f:
            for i in save_dic.keys():
                print('Saving {}'.format(i))
                if i == 'data_scheme.x_indice' and save_dic[i] is None:
                    continue
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
                    elif mem == 'x_indice':
                        try:
                            val = list(f[i][...])
                        except:
                            val = None
                    setattr(data_scheme, mem, val)
                else:
                    if i == 'intercept' or i == 'normalizer':
                        if f[i][...] == 1:
                            setattr(self, i, True)
                        elif f[i][...] == 0:
                            setattr(self, i, False)
                    elif i == 'betahat':
                        setattr(self, i, tf.constant(f[i][:], tf.float32))
        self.data_scheme = data_scheme
                      

class ElasticNetEstimator:
    def __init__(self, data_scheme, alpha, normalizer = False, learning_rate = 0.05, updater = None, lambda_init_dict = None, minimal_load = False):
        '''
        updater is set should be a dict: 
        {
            'updater': updater
            'update_fun': update_fun
        }
        and learning rate will be ignored
        '''
        self.normalizer = normalizer
        self.data_scheme = data_scheme
        self.alpha = alpha
        if minimal_load is False:
            self._init_updater(learning_rate, updater)
            self._init_model(alpha, lambda_init_dict)
    def _init_updater(self, learning_rate, updater):
        '''
        Set up _updater and update_fun
        '''
        if updater is None:
            # use the default: proximal gradient updater
            self._updater = lib_ElasticNet.ProximalUpdater(learning_rate)
            self.update_fun = self._updater.proximal_train_step
        else:
            self._updater = updater['updater']
            self.update_fun = updater['update_fun']
    def _init_model(self, alpha, lambda_init_dict):
        '''
        Initialize the elastic net model.
        Calculate lambda_max using data_init
        '''
        if lambda_init_dict is None:
            lambda_init_dict = {
                'data_init': None, 
                'prefactor_of_lambda_max': 2,
                'lambda_max_over_lambda_min': 1e3,
                'nlambda': 100
            }
        # initialize model
        nx = self.data_scheme.get_num_covariate() + self.data_scheme.get_num_predictor()  
        ny = self.data_scheme.get_num_outcome()
        model_list = []
        for i in range(ny):
            model_lseq = lib_ElasticNet.ElasticNet(nx, alpha, 0)  
            model_list.append(model_lseq)
        # compute lambda sequence 
        if lambda_init_dict['data_init'] is None:
            x_init, y_init = self._lazy_load()   
        else:
            x_init, y_init = lambda_init_dict['data_init']
        lambda_max_list = []
        for i in range(ny):
            lambda_max = util_ElasticNet.get_lambda_max(model_list[i], x_init, y_init[:, i, np.newaxis]) * lambda_init_dict['prefactor_of_lambda_max']
            lambda_max_list.append(lambda_max)
        lambda_seq_list = []
        for i in range(ny):
            lambda_seq = util_ElasticNet.get_lambda_sequence(
                lambda_max_list[i], 
                lambda_max_list[i] / lambda_init_dict['lambda_max_over_lambda_min'], 
                lambda_init_dict['nlambda']
            )
            lambda_seq_list.append(lambda_seq)
        # done and return
        self.model = model_list
        self.lambda_seq = lambda_seq_list
    def _lazy_load(self, max_size = 1000):
        '''
        This is lazy loading.
        If the dataset is repeated.
        It may extract the same element multiple times.
        '''
        dataset_for_load = self.data_scheme.dataset.unbatch().take(max_size)
        if self.normalizer == True:
            normalizer = FullNormalizer(self.data_scheme.get_data_matrix, self.data_scheme.dataset)
        for ele in dataset_for_load.batch(max_size):
            x, y = self.data_scheme.get_data_matrix(ele)
            if self.normalizer == True:
                x = normalizer.apply(x)
            break
        return x, y
    def _concat(self, vec_list):
        '''
        Concatenate a list of 1-dim numpy array
        '''
        tmp = vec_list[0]
        for i in range(1, len(vec_list)):
            tmp = np.concatenate((tmp, vec_list[i]), axis = 0)
        return tmp
    def _out_dim_covar(self):
        if self.data_scheme is None:
            return None
        else:
            n_predictor = self.data_scheme.get_num_predictor()
            n_covariate = self.data_scheme.get_num_covariate()
            n_outcome = self.data_scheme.get_num_outcome()
            x_dim = n_predictor
            c_dim = n_covariate
            y_dim = n_outcome
            if self.intercept is True:
                c_dim += 1  
            return x_dim, c_dim, y_dim
    
    def solve_pt(self, abs_z_cutoffs = [ 0.1, 1 ], rcond = 1e-10, scaling = True):
        '''
        Solve P+T with self.alpha being interpreted as R2 cutoff.
        Step 1: Run linear regression y ~ gene_i + covars for one gene at a time
        to obtain z-score for each gene.
        Step 2: Calculate correlation between genes.
        Step 3: Performing P+T.
        '''
        self.lambda_seq = [ abs_z_cutoffs.copy() for i in range(self.data_scheme.get_num_outcome()) ]
        svd = SVDInstance(rcond)
        print('Don\'t support normalization.')
        self.normalizer = False
        print('Add intercept.')
        self.intercept = True
        xdim, cdim, ydim = self._out_dim_covar()
        ctc = tf.Variable(initial_value = tf.zeros([cdim, cdim], tf.float32))
        ctx = tf.Variable(initial_value = tf.zeros([cdim, xdim], tf.float32))
        cty = tf.Variable(initial_value = tf.zeros([cdim, ydim], tf.float32))
        n_processed = 0
        mean_x = tf.Variable(initial_value = tf.zeros([xdim], tf.float32))
        for ele in self.data_scheme.dataset:
            x, y = self.data_scheme.get_data_matrix(ele, only_x = True)
            c, _ = self.data_scheme.get_data_matrix(ele, only_covar = True)
            # c = self._prep_for_intercept(c)
            c = tf.concat((tf.ones([c.shape[0], 1], tf.float32), c), axis = 1)
            n_new = x.shape[0]
            n_old = n_processed
            # val_old_mean * (n_old / (n_old + n_new)) + val_new_sum / (n_old + n_new) 
            if scaling is True:
                f_old = n_old / (n_old + n_new)
                f_new = 1 / (n_old + n_new)
            else:
                f_old = 1
                f_new = 1
            ctc.assign(tf.add(
                    tf.multiply(ctc, f_old), 
                    tf.matmul(tf.multiply(tf.transpose(c), f_new), c)
                )
            )
            mean_x.assign(tf.add(
                    tf.multiply(mean_x, f_old),
                    tf.multiply(tf.reduce_sum(x, axis=0), f_new)
                )
            )
            # ctx.assign(tf.add(
            #         tf.multiply(ctx, f_old), 
            #         tf.matmul(tf.multiply(tf.transpose(c), f_new), x)
            #     )
            # )
            cty.assign(tf.add(
                    tf.multiply(cty, f_old),
                    tf.matmul(tf.multiply(tf.transpose(c), f_new), y)
                )
            )
            n_processed += n_new
        
        # svd on ctc
        svd.solve(ctc)
        
        # calculate beta hat
        # covar_betahat_x = tf.matmul(
        #     tf.matmul(
        #         tf.matmul(
        #             svd.v, 
        #             tf.linalg.tensor_diag(tf.math.reciprocal_no_nan(svd.s))
        #         ), 
        #         tf.transpose(svd.u)
        #     ), 
        #     ctx
        # )
        covar_betahat = tf.matmul(
            tf.matmul(
                tf.matmul(
                    svd.v,
                    tf.linalg.tensor_diag(tf.math.reciprocal_no_nan(svd.s))
                ),
                tf.transpose(svd.u)
            ),
            cty
        )
        
        x2 = tf.Variable(initial_value = tf.zeros([xdim], tf.float32))
        # x2o = tf.Variable(initial_value = tf.zeros([xdim], tf.float32))
        xty = tf.Variable(initial_value = tf.zeros([xdim, ydim], tf.float32))
        y2 = tf.Variable(initial_value = tf.zeros([ydim], tf.float32))
        xtx = tf.Variable(initial_value = tf.zeros([xdim, xdim], tf.float32))
        n_processed = 0
        for ele in self.data_scheme.dataset:
            x, y = self.data_scheme.get_data_matrix(ele, only_x = True)
            c, _ = self.data_scheme.get_data_matrix(ele, only_covar = True)
            c = tf.concat((tf.ones([c.shape[0], 1], tf.float32), c), axis = 1)
            x = x - tf.broadcast_to(mean_x, [x.shape[0], xdim])
            xtx.assign(tf.add(
                    tf.multiply(xtx, f_old), 
                    tf.matmul(tf.multiply(tf.transpose(x), f_new), x)
                )
            )
            # x2o.assign(tf.add(
            #         tf.multiply(x2o, f_old),
            #         tf.multiply(
            #             tf.reduce_sum(tf.math.square(x), axis = 0),
            #             f_new
            #         )
            #     )
            # )
            # x = x - tf.matmul(
            #     c,
            #     covar_betahat_x,
            # )
            y = y - tf.matmul(
                c,
                covar_betahat,
            )
            x2.assign(tf.add(
                    tf.multiply(x2, f_old), 
                    tf.multiply(
                        tf.reduce_sum(tf.math.square(x), axis = 0), 
                        f_new
                    )
                )
            )
            y2.assign(tf.add(
                    tf.multiply(y2, f_old), 
                    tf.multiply(
                        tf.reduce_sum(tf.math.square(y), axis = 0), 
                        f_new
                    )
                )
            )
            xty.assign(tf.add(
                    tf.multiply(xty, f_old), 
                    tf.matmul(tf.multiply(tf.transpose(x), f_new), y)
                )
            )
            n_processed += n_new
        weights = tf.einsum(
            'j,jk->jk', 
            tf.math.reciprocal_no_nan(x2), 
            xty
        )
        s2 = tf.broadcast_to(y2, [xdim, ydim]) - 2 * weights * xty + (weights ** 2) * tf.transpose(tf.broadcast_to(x2, [ydim, xdim]))
        sd = tf.sqrt(s2 / tf.transpose(tf.broadcast_to(x2, [ydim, xdim])) / n_processed)
        corr = tf.einsum(
            'j,jk->jk', 
            tf.math.reciprocal_no_nan(tf.math.sqrt(x2)), 
            xtx
        )
        corr = tf.einsum(
            'k,jk->jk', 
            tf.math.reciprocal_no_nan(tf.math.sqrt(x2)), 
            corr
        )
        zscores = weights / sd
        abs_zs = tf.math.abs(zscores)
        
        weights_pt = weights.numpy()
        corr_n = corr.numpy()
        abs_zs_n = abs_zs.numpy()
        for trait_idx in range(abs_zs_n.shape[1]):
            sort_zs_order = np.argsort(-abs_zs_n[:, trait_idx])
            selected_dict = { idx: 0 for idx in sort_zs_order }
            # 0: not yet settled down
            # 1: included
            # -1: discarded
            for curr_idx in sort_zs_order:
                if selected_dict[curr_idx] == -1:
                    continue
                elif selected_dict[curr_idx] == 0:
                    selected_dict[curr_idx] = 1
                    for possible_idx in range(xdim):
                        if selected_dict[possible_idx] == 0:
                            if (corr_n[curr_idx, possible_idx] ** 2) >= self.alpha:
                                selected_dict[possible_idx] = -1
                else:
                    raise ValueError('Something wrong: processing')
            discarded_idx = []
            for idx in range(xdim):
                if selected_dict[idx] == 0:
                    raise ValueError('Something wrong: post') 
                elif selected_dict[idx] == -1:
                    discarded_idx.append(idx) 
            weights_pt[discarded_idx, trait_idx] = 0
            
        n_covar = self.data_scheme.get_num_covariate()
        n_pred = self.data_scheme.get_num_predictor()
        n_model = len(self.model)
        n_lambda = len(self.lambda_seq[0])
        beta_hat = np.empty((n_pred, n_model, n_lambda))
        covar_hat = np.zeros((n_covar, n_model, n_lambda))
        intercept_hat = np.zeros((1, n_model, n_lambda))
        for mi, ml in enumerate(self.lambda_seq):
            tmp = weights_pt[:, mi].copy()
            for li, ll in enumerate(ml):
                tmp2 = tmp.copy()
                tmp2[abs_zs[:, mi] <= ll] = 0 
                beta_hat[:, mi, li] = tmp2
                covar_hat[:, mi, li] = covar_betahat[1:, mi]
                intercept_hat[:, mi, li] = covar_betahat[0, mi]
        self.beta_hat_path = tf.constant(beta_hat, tf.float32)
        self.covar_hat_path = tf.constant(covar_hat[1:, :], tf.float32)
        self.intercept_path = tf.constant(intercept_hat[0, :], tf.float32)
        
    def solve(self, checker, nepoch = 10, logging = None, x_check = None, y_check = None):
        '''
        Solve for a sequence of lambda and return a sequence of betahat (for x, covar, and intercept) correspondingly and the objective captured by checker (a list with num_outcomes number of checkers).
        From lambda_max to lambda_min, it solves a sequence of Elastic Net models. 
        At each inner loop, it runs in batch (the batch size is determined by dataset). 
        For each epoch (one scanning through the data), checker will evaluate objective on (x_check, y_check) or last batch (x, y).
        And determine if to stop by looking at the objective sequence (the rule is specified in stop_rule).
        '''
        if self.normalizer == True:
            if logging is not None:
                logging.info('start norm')
            normalizer = FullNormalizer(self.data_scheme.get_data_matrix, self.data_scheme.dataset)
            if logging is not None:
                logging.info('end norm')
        n_covar = self.data_scheme.get_num_covariate()
        n_pred = self.data_scheme.get_num_predictor()
        n_model = len(self.model)
        n_lambda = len(self.lambda_seq[0])
        beta_hat = np.empty((n_pred, n_model, n_lambda))
        covar_hat = np.empty((n_covar, n_model, n_lambda))
        intercept_hat = np.empty((1, n_model, n_lambda))
        checker_summary = [ [] for the_jth_model in range(n_model) ]
        loop_summary = [ [] for the_jth_model in range(n_model) ]
        counter = 0
        outer_counter = 0
        for the_ith_lambda in range(n_lambda):
            
            # loop over models
            for the_jth_model in range(n_model):
                self.model[the_jth_model].update_lambda(self.lambda_seq[the_jth_model][the_ith_lambda])
                checker[the_jth_model].reset()
            # end looping
            
            converged_models = [ False for the_jth_model in range(n_model) ]
            for ele in self.data_scheme.dataset.repeat(nepoch):
                x, y = self.data_scheme.get_data_matrix(ele)
                if self.normalizer == True:
                    x = normalizer.apply(x)
                step_size = x.shape[0]
                
                # loop over models
                update_status = []
                any_update = 0
                epoch_idx = 0
                for the_jth_model in range(n_model):
                    if converged_models[the_jth_model] is True:
                        continue
                    any_update = 1
                    self.update_fun(self.model[the_jth_model], x, y[:, the_jth_model, np.newaxis])
                    update_status.append(checker[the_jth_model].update(step_size))
                    if update_status[-1] == 0:
                        epoch_idx = checker[the_jth_model].epoch_counter
                        if x_check is None or y_check is None:
                            obj_check = self.model[the_jth_model].objective(x, y[:, the_jth_model, np.newaxis])[0]
                        else:
                            obj_check = self.model[the_jth_model].objective(x_check, y_check[:, the_jth_model, np.newaxis])[0]
                        checker[the_jth_model].record(update_status[-1], obj_check)
                        if checker[the_jth_model].ifstop() == True:
                            converged_models[the_jth_model] = True
                # end looping
                
                if update_status[0] == 0:
                    # gone through one epoch
                    if logging is not None:
                        logging.info('Gone through outer loop {} / {} and inner loop {} epoch'.format(outer_counter + 1, n_lambda, epoch_idx))
                if sum(converged_models) == n_model:
                    break
                    
            # loop over models
            for the_jth_model in range(n_model):
                beta_hat[:, the_jth_model, counter] = self.model[the_jth_model].A[:n_pred, 0]
                covar_hat[:, the_jth_model, counter] = self.model[the_jth_model].A[n_pred:, 0]
                intercept_hat[:, the_jth_model, counter] = self.model[the_jth_model].b[0]
                checker_summary[the_jth_model].append(np.array(checker[the_jth_model].criteria_summary))
                loop_summary[the_jth_model].append(checker[the_jth_model].epoch_counter)
            # end looping
            
            outer_counter += 1
            counter += 1
            
        self.beta_hat_path = tf.constant(beta_hat, tf.float32)
        self.covar_hat_path = tf.constant(covar_hat, tf.float32)
        self.intercept_path = tf.constant(intercept_hat, tf.float32)
        
        # loop over models
        return_dic = { 'obj': [], 'niter': [] }
        for the_jth_model in range(n_model):
            return_dic['obj'].append(self._concat(checker_summary[the_jth_model]))
            return_dic['niter'].append(np.array(loop_summary[the_jth_model]))
        # end looping
        
        return return_dic
    def predict(self, dataset, beta, covar, intercept):
        '''
        We assume dataset has the same data_scheme as self.data_scheme.
        y_pred = ((intercept), x, covar) %*% betahat
        It returns y, ypred as numpy array
        beta, covar, intercept should be 2-dim (n_var, k_models)
        '''
        Amat = tf.constant(np.concatenate((beta, covar), axis = 0), tf.float32)
        bmat = tf.constant(intercept, tf.float32)
        y_ = []
        y_pred_ = []
        if self.normalizer == True:
            normalizer = FullNormalizer(self.data_scheme.get_data_matrix, dataset)
        for ele in dataset:
            x, y = self.data_scheme.get_data_matrix(ele)
            if self.normalizer == True:
                x = normalizer.apply(x)
            y_pred_.append(
                tf.math.add(
                    tf.einsum('ij,jkq->ikq', x, Amat),  # tf.matmul(x, Amat),
                    bmat
                )
            )
            y_.append(y)
        y_pred_ = self._reshape_y(y_pred_)
        y_ = self._reshape_y(y_)
        return {'y_pred': y_pred_, 'y': y_}
    def predict_x(self, dataset, beta):
        '''
        We assume dataset has the same data_scheme as self.data_scheme.
        y_pred = x %*% betahat_x
        It returns y, ypred as numpy array
        '''
        Amat = tf.constant(beta, tf.float32)
        y_ = []
        y_pred_ = []
        if self.normalizer == True:
            scheme_func = functools.partial(self.data_scheme.get_data_matrix, only_x = True)
            normalizer = FullNormalizer(scheme_func, dataset)
        for ele in dataset:
            x, y = self.data_scheme.get_data_matrix(ele, only_x = True)
            if self.normalizer == True:
                x = normalizer.apply(x)
            y_pred_.append(tf.einsum('ij,jkq->ikq', x, Amat))
            y_.append(y)
        y_pred_ = self._reshape_y(y_pred_)
        y_ = self._reshape_y(y_)
        print(y_pred_.shape)
        return {'y_pred_from_x': y_pred_, 'y': y_} 
    def _reshape_y(self, y_list):
        return np.concatenate(y_list, axis = 0)
    def minimal_save(self, filename):
        '''
        Perform minimal save, which saves the minimal things needed for prediction. 
        And it loses a huge amount of information but they may not be of interest anyway.
        They are: 
        1) lambda_seq
        2) beta_hat_path, covar_hat_path, and intercept_path
        3) all members but dataset in data_scheme;
        4) intercept
        5) batch normalization shuffle
        '''
        save_dic = {}
        save_dic['lambda_seq'] = np.array(self.lambda_seq)
        save_dic['beta_hat_path'] = self.beta_hat_path.numpy()
        save_dic['covar_hat_path'] = self.covar_hat_path.numpy()
        save_dic['intercept_path'] = self.intercept_path.numpy()
        save_dic['normalizer'] = self.normalizer * 1
        save_dic['alpha'] = self.alpha
        for i in self.data_scheme.__dict__.keys():
            if i != 'dataset':
                save_dic['data_scheme.' + i] = getattr(self.data_scheme, i)
            else:
                save_dic['data_scheme.' + i] = b'save_mode'
        with h5py.File(filename, 'w') as f:
            for i in save_dic.keys():
                print('Saving {}'.format(i))
                if i == 'data_scheme.x_indice' and save_dic[i] is None:
                    continue
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
                    elif mem == 'x_indice':
                        try:
                            val = list(f[i][...])
                        except:
                            val = None
                    setattr(data_scheme, mem, val)
                else:
                    if i == 'normalizer':
                        if f[i][...] == 1:
                            setattr(self, i, True)
                        elif f[i][...] == 0:
                            setattr(self, i, False)
                    elif i in ('beta_hat_path', 'covar_hat_path', 'intercept_path'):
                        setattr(self, i, tf.constant(f[i][:], tf.float32))
                    elif i == 'lambda_seq':
                        lambda_mat = f[i][:]
                        self.lambda_seq = [ lambda_mat[i, :] for i in range(lambda_mat.shape[0]) ]
                    elif i == 'alpha':
                        self.alpha = f[i][...]
        self.data_scheme = data_scheme

