import tensorflow as tf

from tensorflow.keras import Model

class cnnPTRS(Model):
    def __init__(self, struct_ordered_dict, num_outcomes):
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
        super(cnnPTRS, self).__init__()
        self.__init_cnn_layers(struct_ordered_dict)
        self.linear_predictor = tf.keras.layers.Dense(num_outcomes, activation='linear')
    def call(self, inputs):
        x1 = inputs[0]
        for l in self.serialized_layers:
            x1 = getattr(self, l)(x1)
        x2 = inputs[1]
        x = layers.concatenate([x1, x2])
        return self.linear_predictor(x)
    def __init_cnn_layers(self, struct_ordered_dict):
        self.serialized_layers = []
        for layer_name in struct_ordered_dict.keys():
            layer_dict = struct_ordered_dict[layer_name]
            if 'conv' not in layer_dict:
                continue
            else:
                setattr(self, f'conv_{layer_name}', tf.keras.layers.Conv1D(**layer_dict['conv']))
                self.serialized_layers.append(f'conv_{layer_name}')
                if 'maxpool' in layer_dict:
                    setattr(self, f'maxpool_{layer_name}', tf.keras.layers.MaxPool1D(**layer_dict['maxpool']))
                    self.serialized_layers.append(f'conv_{layer_name}')
                if 'dropout' in layer_dict:
                    setattr(self, f'maxpool_{layer_name}', tf.keras.layers.Dropout(**layer_dict['dropout']))
                    self.serialized_layers.append(f'conv_{layer_name}')
        self.final_flatten = tf.keras.layers.Flatten()
        self.serialized_layers.append('final_flatten')
    