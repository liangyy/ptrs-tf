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
                    x_ = tf.keras.layers.Conv1D(**layer_dict['conv'])(inputx)
                    counter = 1
                else:
                    x_ = tf.keras.layers.Conv1D(**layer_dict['conv'])(x_)
                if 'maxpool' in layer_dict:
                    x_ = tf.keras.layers.MaxPool1D(**layer_dict['maxpool'])(x_)
                if 'dropout' in layer_dict:
                    x_ = tf.keras.layers.Dropout(**layer_dict['dropout'])(x_)
        x_ = tf.keras.layers.Flatten()(x_)
        x_n_covar_ = tf.keras.layers.concatenate([x_, covar_])
        outputy = tf.keras.layers.Dense(self.num_outcomes, activation = 'linear')(x_n_covar_)
        self.model = tf.keras.Model(inputs = [inputx, covar_], outputs = outputy)
        


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
