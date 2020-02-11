import tensorflow as tf

class cnnPTRS:
    def __init__(self, struct_ordered_dict, num_x, num_outcomes, num_covar):
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
        # super(cnnPTRS, self).__init__()
        self.num_x = num_x
        self.num_outcomes = num_outcomes
        self.num_covar = num_covar
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
    def _mse_loss_tf(self, y, yp):
        return tf.reduce_mean(tf.math.pow(y - yp, 2))
    def _mean_cor_tf(self, y, yp):
        return tf.reduce_mean(self._cor_tf(y, yp))
    def _cor_tf(self, y, yp):
        '''
        cov_xy / sqrt(var_x * var_y)
        '''
        o1, o2, o3 = self.__var_x_y_all_tf(y, yp)
        return tf.divide(o3, tf.sqrt( tf.multiply(o1, o2) ))
    def __var_x_y_all_tf(self, x, y):
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
    def _train_one_step(self, optimizer, x, y):
        with tf.GradientTape() as tape:
            y_, _ = self.model(x, training = True)
            loss = self._mse_loss_tf(y, y_)
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    def predict(self, inputs): 
        y, _ = self.model(inputs, training = False)   
        return y
    def predict_x(self, inputs):
        _, y = self.model(inputs, training = False)   
        return y
    @tf.function
    def train(self, optimizer, data_scheme, num_epoch, inputs_valid, outcomes_valid):
        step = 0
        loss = 0.0
        valid_accuracy = 0.0
        for ele in dataset.repeat(num_epoch):
            inputs, y = data_scheme.get_data_matrix(ele)
            # inputs = [tf.expand_dims(x, axis = 2), y[:, self.num_outcomes:]]
            # y = y[:, :self.num_outcomes]
            step += 1
            loss = train_one_step(model, optimizer, inputs, y)
            if step % 10 == 0:
                yp = predict(model, inputs_valid)
                valid_accuracy = self._mean_cor_tf(yp, y_valid)
                tf.print('Step', step, ': loss', loss, '; validation-accuracy:', valid_accuracy)
        return step, loss, valid_accuracy
    


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
#                     self.serialized_layers.append(f'conv_{layer_name}')
#         self.final_flatten = tf.keras.layers.Flatten()
#         self.serialized_layers.append('final_flatten')
# 
