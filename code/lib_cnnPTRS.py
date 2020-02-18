import tensorflow as tf
from lib_LinearAlgebra import FullNormalizer, DataScheme
import sys
import h5py

class mlpPTRS(kerasPTRS):
    def __init__(self, struct_ordered_dict, data_scheme, temp_path, normalizer = False, minimal_load = False):
        '''
        For MLP architecture:
        struct_ordered_dict:
            unit1:
                kwargs
            unit2:
                ...
        All units are Dense()
        Overall architecture:
            - - - - - - if struct_ordered_dict is None
            |         |
            x1 -MLP-> m1 --|
                           +-- linear predictor -> y
                      x2 --|
        '''
        super().__init__(data_scheme, temp_path, normalizer = False, minimal_load = False)
        self.__init_mlp_layers(struct_ordered_dict)
    def __init_mlp_layers(self, struct_ordered_dict):
        inputx = tf.keras.Input(shape = (self.num_x, 1))
        covar_ = tf.keras.Input(shape = (self.num_covar))
        if struct_ordered_dict is not None:
            counter = 0
            for layer_name in struct_ordered_dict.keys():
                kwargs_i = struct_ordered_dict[layer_name]
                if counter == 0:
                    x_ = tf.keras.layers.Dense(**kwargs_i, name = f'{layer_name}_dense')(inputx)
                    counter = 1
                else:
                    x_ = tf.keras.layers.Dense(**kwargs_i, name = f'{layer_name}_dense')(x_)   
            x_ = tf.keras.layers.Flatten()(x_)
        else: 
            x_ = tf.keras.layers.Flatten()(inputx)
        output_x_ = tf.keras.layers.Dense(self.num_outcomes, activation = 'linear', use_bias = False, name = 'ptrs_dense')(x_)
        output_covar_ = tf.keras.layers.Dense(self.num_outcomes, activation = 'linear', name = 'covar_dense')(covar_)
        outputy = tf.keras.layers.Add()([output_x_, output_covar_])
        self.model = tf.keras.Model(inputs = [inputx, covar_], outputs = [outputy, output_x_])
class cnnPTRS(kerasPTRS):
    def __init__(self, struct_ordered_dict, data_scheme, temp_path, normalizer = False, minimal_load = False):
        '''
        For CNN architecture
        struct_ordered_dict:
            unit1:
                conv:
                    kwargs
                maxpool:
                    kwargs
                dropout:
                    kwargs
            unit2:  
                ...
        Overall architecture:
            x1 -CNN-> m1 --|
                           +-- linear predictor -> y
                      x2 --|
        '''
        super().__init__(data_scheme, temp_path, normalizer = False, minimal_load = False)
        self.__init_cnn_layers(struct_ordered_dict)
    def __init_cnn_layers(self, struct_ordered_dict):
        inputx = tf.keras.Input(shape = (self.num_x, 1))
        covar_ = tf.keras.Input(shape = (self.num_covar))
        counter = 0
        for layer_name in struct_ordered_dict.keys():
            layer_dict = struct_ordered_dict[layer_name]
            if 'conv' not in layer_dict:
                continue
            else:
                if counter == 0:
                    x_ = tf.keras.layers.Conv1D(**layer_dict['conv'], name = f'{layer_name}_conv')(inputx)
                    counter = 1
                else:
                    x_ = tf.keras.layers.Conv1D(**layer_dict['conv'], name = f'{layer_name}_conv')(x_)
                if 'maxpool' in layer_dict:
                    x_ = tf.keras.layers.MaxPool1D(**layer_dict['maxpool'], name = f'{layer_name}_maxpool')(x_)
                if 'dropout' in layer_dict:
                    x_ = tf.keras.layers.Dropout(**layer_dict['dropout'], name = f'{layer_name}_dropout')(x_)
        x_ = tf.keras.layers.Flatten()(x_)
        output_x_ = tf.keras.layers.Dense(self.num_outcomes, activation = 'linear', use_bias = False, name = 'ptrs_dense')(x_)
        output_covar_ = tf.keras.layers.Dense(self.num_outcomes, activation = 'linear', name = 'covar_dense')(covar_)
        outputy = tf.keras.layers.Add()([output_x_, output_covar_])
        self.model = tf.keras.Model(inputs = [inputx, covar_], outputs = [outputy, output_x_])
class kerasPTRS:
    def __init__(self, data_scheme, temp_path, normalizer = False, minimal_load = False):
        '''
        temp_path: to save best model during training
        '''
        self.temp_path = temp_path
        if minimal_load is False:
            self.normalizer = normalizer
            self.data_scheme = data_scheme
            self.temp_path = temp_path
            self.__init_from_data_scheme()
    def __init_from_data_scheme(self):
        self.num_x = self.data_scheme.get_num_predictor()
        self.num_outcomes = self.data_scheme.get_num_outcome()
        self.num_covar = self.data_scheme.get_num_covariate()
    def _mse_loss_tf(self, y, yp):
        return tf.reduce_mean(tf.math.pow(y - yp, 2))
    def _mean_cor_tf(self, y, yp):
        return tf.reduce_mean(self._cor_tf(y, yp))
    def _cor_tf(self, y, yp):
        '''
        cov_xy / sqrt(var_x * var_y)
        '''
        o1, o2, o3 = self._var_x_y_all_tf(y, yp)
        return tf.divide(o3, tf.sqrt( tf.multiply(o1, o2) ))
    def _var_x_y_all_tf(self, x, y):
        '''
        var_x_y = mean( ( x - mean(x) ) * ( y - mean(y) ) ) 
        '''
        mx = tf.reduce_mean(x, axis = 0)
        my = tf.reduce_mean(y, axis = 0)
        diff_x_mx = x - mx
        diff_y_my = y - my
        o1 = tf.reduce_mean( tf.multiply(diff_x_mx, diff_x_mx), axis = 0 )
        o2 = tf.reduce_mean( tf.multiply(diff_y_my, diff_y_my), axis = 0 )
        o3 = tf.reduce_mean( tf.multiply(diff_x_mx, diff_y_my), axis = 0 )
        return o1, o2, o3
    # @tf.function
    def _train_one_step(self, optimizer, x, y, var_list):
        with tf.GradientTape() as tape:
            y_, _ = self.model(x, training = True)
            loss = self._mse_loss_tf(y, y_)
        grads = tape.gradient(loss, var_list)  # self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, var_list))  # self.model.trainable_variables))
        return loss
    def _predict(self, inputs): 
        y, _ = self.model(inputs, training = False)   
        return y
    def _predict_x(self, inputs):
        _, y = self.model(inputs, training = False)   
        return y
    def _ele_unpack(self, ele):
        inputs, y = self.data_scheme.get_data_matrix_x_in_cnn(ele)
        if self.normalizer == True:
            normalizer_ = FullNormalizer(self.data_scheme.get_data_matrix_x_in_cnn, ele, tensor = True)
            inputs = normalizer_.apply(inputs)
        return inputs, y
    def predict(self, ele):
        inputs, y = self._ele_unpack(ele)
        return self._predict(inputs), y
    def predict_x(self, ele):
        inputs, y = self._ele_unpack(ele)
        return self._predict_x(inputs), y   
    def prep_train(self, ele_valid):
        if self.normalizer == True:
            normalizer = FullNormalizer(self.data_scheme.get_data_matrix_x_in_cnn, self.data_scheme.dataset)
            normalizer_valid = FullNormalizer(self.data_scheme.get_data_matrix_x_in_cnn, ele_valid, tensor = True)
            return normalizer, normalizer_valid
        else:
            return None, None
    def train_func(self, var_list = None):
        if var_list is None:
            v = self.model.trainable_variables
        else:
            v = var_list
        @tf.function
        def train(self, optimizer, num_epoch, ele_valid, logging, normalizer = None, normalizer_valid = None, var_list = v, ele_insample = None):
            step = 0
            loss = 0.0
            valid_accuracy = 0.0
            valid_accuracy_x = 0.0
            insample_accuracy_x = 0.0
            best_v_accuracy_x = 0.0
            # work-around so that tf.function decoration works (.shape is not working in current tf2 version)
            # if self.normalizer == True:
            #     normalizer = FullNormalizer(self.data_scheme.get_data_matrix_x_in_cnn, self.data_scheme.dataset)
            #     normalizer_valid = FullNormalizer(self.data_scheme.get_data_matrix_x_in_cnn, ele_valid, tensor = True)
            inputs_valid, y_valid = self.data_scheme.get_data_matrix_x_in_cnn(ele_valid)
            if self.normalizer == True:
                inputs_valid = normalizer_valid.apply(inputs_valid)
            if ele_insample is not None:
                inputs_insample, y_insample = self.data_scheme.get_data_matrix_x_in_cnn(ele_insample)
                if self.normalizer == True:
                    inputs_insample = normalizer_valid.apply(inputs_insample)
            ypx = self._predict_x(inputs_valid)
            best_v_accuracy_x = self._mean_cor_tf(ypx, y_valid)
            for epoch in range(num_epoch):
                for ele in self.data_scheme.dataset:
                    inputs, y = self.data_scheme.get_data_matrix_x_in_cnn(ele)
                    if self.normalizer == True:
                        inputs = normalizer.apply(inputs)
                    step += 1
                    loss = self._train_one_step(optimizer, inputs, y, var_list)
                    
                yp = self._predict(inputs_valid)
                ypx = self._predict_x(inputs_valid)
                valid_accuracy = self._mean_cor_tf(yp, y_valid)
                valid_accuracy_x = self._mean_cor_tf(ypx, y_valid)
                if ele_insample is not None:
                    ypx_in = self._predict_x(inputs_insample)
                    insample_accuracy_x = self._mean_cor_tf(ypx_in, y_insample)
                # tf.print('@@@@ Epoch', epoch, ': loss', loss, '; validation-accuracy:', valid_accuracy, '; validation-accurary-x', valid_accuracy_x, '; insample-accuracy-x', insample_accuracy_x, output_stream = log_path)
                tf.py_function(self._print, ['@@@@ Epoch', epoch, ': loss', loss, '; validation-accuracy:', valid_accuracy, '; validation-accurary-x', valid_accuracy_x, '; insample-accuracy-x', insample_accuracy_x], [])
                if best_v_accuracy_x < valid_accuracy_x:
                    tf.py_function(self._print, ['@@@@ Saving model after current epoch', epoch], [])
                    best_v_accuracy_x = valid_accuracy_x
                    # outfile = self.temp_path
                    tf.py_function(self._model_save, [], [])
            return step, loss, valid_accuracy, valid_accuracy_x
        return train
    def add_logger(self, logger):
        self.logging = logger
    def _print(self, *args):
        o = []
        for i in args:
            i = i.numpy()
            # breakpoint()
            if isinstance(i, bytes):
                i = i.decode()
                # breakpoint()
            else:
                i = str(i)
            o.append(i)
        # breakpoint()
        self.logging.info(' '.join(o))
    def _model_save(self):
        self.model.save(self.temp_path)
    def minimal_save(self, filename, save_curr = True):
        '''
        If save_curr is False, save the model from self.temp_path
        Perform minimal save, which saves the minimal things needed for prediction. 
        They are: 
        1) CNN model; 
        2) all members but dataset in data_scheme;
        3) normalizer value
        '''
        save_dic = {}
        save_dic['keras_model_path'] = 'keras-model-save' + filename
        if save_curr is True:
            self.model.save(save_dic['keras_model_path'])
        else:
            model_ = tf.keras.models.load_model(self.temp_path)
            model_.save(save_dic['keras_model_path'])
            del model_
        save_dic['normalizer'] = self.normalizer * 1
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
        Note that it may not be a perfect load.
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
                    elif i == 'keras_model_path':
                        self.model = tf.keras.models.load_model(f[i][...])
        self.data_scheme = data_scheme
        # self.__init_cnn_layers(struct_ordered_dict)
                      


# class cnnPTRS(Model):
#     def __init__(self, struct_ordered_dict, num_outcomes):
#         '''
#         For CNN architecture
#         struct_ordered_dict:
#             unit1:
#                 conv:
#                     kwargs
#                 maxpool:
#                     kwargs
#                 dropout:
#                     kwargs
#             unit2:  
#                 ...
#         Overall architecture:
#             x1 -CNN-> m1 --|
#                            +-- linear predictor -> y
#                       x2 --|
#         '''
#         super(cnnPTRS, self).__init__()
#         self.__init_cnn_layers(struct_ordered_dict)
#         self.linear_predictor = tf.keras.layers.Dense(num_outcomes, activation='linear')
#     def call(self, inputs):
#         x1 = inputs[0]
#         for l in self.serialized_layers:
#             x1 = getattr(self, l)(x1)
#         x2 = inputs[1]
#         x = layers.concatenate([x1, x2])
#         return self.linear_predictor(x)
#     def __init_cnn_layers(self, struct_ordered_dict):
#         self.serialized_layers = []
#         for layer_name in struct_ordered_dict.keys():
#             layer_dict = struct_ordered_dict[layer_name]
#             if 'conv' not in layer_dict:
#                 continue
#             else:
#                 setattr(self, f'conv_{layer_name}', tf.keras.layers.Conv1D(**layer_dict['conv']))
#                 self.serialized_layers.append(f'conv_{layer_name}')
#                 if 'maxpool' in layer_dict:
#                     setattr(self, f'maxpool_{layer_name}', tf.keras.layers.MaxPool1D(**layer_dict['maxpool']))
#                     self.serialized_layers.append(f'conv_{layer_name}')
#                 if 'dropout' in layer_dict:
#                     setattr(self, f'maxpool_{layer_name}', tf.keras.layers.Dropout(**layer_dict['dropout']))
