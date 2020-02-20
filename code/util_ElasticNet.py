import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import h5py
import re
import util_Stats

# elastic net math related
def get_lambda_max(model, x, y):
    '''
    `model` is object of ElasticNet class
    return lambda_max given dataset x, y and alpha of ElasticNet model 
    '''
    a = model.A
    b = model.b
    model.A.assign(tf.zeros(a.shape))
    model.b.assign(tf.ones(b.shape), tf.reduce_mean(y))
    with tf.GradientTape() as tape:
        obj, loss = model.proximal_obj(x, y)
    grad = tape.gradient(obj, [model.proximal_variables, model.not_prox_variables])
    l1_max = tf.reduce_max(tf.abs(grad[0]))
    lambda_max = l1_max / model.alpha
    return lambda_max
    
def get_lambda_sequence(lambda_max, lambda_min, nlambda):
    '''
    return a sequence with equal space in log scale (numpy array) 
    '''
    seq = np.linspace(np.log(lambda_max), np.log(lambda_min), num = nlambda)
    return np.exp(seq)
# END --

# train_step for built-in optimizer
def train_step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        obj, loss = model.objective(x, y)
    grad = tape.gradient(obj, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return loss, obj
# END --

# data I/O
def load_hdf5_as_dataset(filename_list, dataset_list, batch_size, num_epochs, shuffle = None, take = None, inv_norm_y = False, covar_indice = None, preset_y = None):
    '''
    `filename_list` and `dataset_list` should be file path and dataset name of x and y
    It maps all numbers to tf.float32 using tf.cast
    The value in `shuffle` specifies the `buffer_size` 
    '''
    # get to know the size of dataset 
    h5handle = h5py.File(filename_list[0], 'r')
    size = h5handle[dataset_list[0]].shape[0]
    h5handle.close()
    # load dataset ad TF Dataset
    X = tfio.IODataset.from_hdf5(filename_list[0], dataset_list[0])
    if inv_norm_y is False and preset_y is None:
        y = tfio.IODataset.from_hdf5(filename_list[1], dataset_list[1])
    elif preset_y is not None:
        y = preset_y
    elif inv_norm_y is True:
        with h5py.File(filename_list[1], 'r') as f:
            name_y = re.sub('^/', '', dataset_list[1])
            y = f[name_y][:]
            y = util_Stats.inv_norm_col(y, exclude_idx = covar_indice)
        y = tf.data.Dataset.from_tensor_slices(y)
    if take is not None:
        X = X.take(take)
        y = y.take(take)
    dataset = tf.data.Dataset.zip((X, y))
    dataset = dataset.map(lambda x , y : (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
    if shuffle is not None:
        dataset = dataset.shuffle(shuffle)
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    dataset = dataset.prefetch(1)
    return dataset, size
def load_hdf5_as_tensor(filename, dataset_list):
    test_data = tfio.IOTensor.from_hdf5(filename_list)
    out_list = []
    for dataset in dataset_list:
        out_list.append(tf.cast(test_data(dataset).to_tensor(), tf.float32))
    return out_list
def load_predixcan_hdf5(filename, dataset_sample, dataset_gene, dataset_matrix, training_samples, tensor_samples_list):
    '''
    training data will be loaded as TF2 Dataset.
    tensor data will be loaded as TF2 Tensor (in memory). 
    '''
    h5handle = h5py.File(filename, 'r')
    sample_names = h5handle[dataset_sample][:]
    
    X = tfio.IODataset.from_hdf5(filename, dataset_list[0])
    return d
# END --

