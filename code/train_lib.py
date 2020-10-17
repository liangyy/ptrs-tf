import os
import util_ElasticNet, lib_LinearAlgebra, util_hdf5, lib_ElasticNet, lib_Checker, util_Stats, util_misc
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py, yaml, functools


def prep_dataset_from_hdf5(input_hdf5, data_scheme_yaml, 
batch_size, logging, against_hdf5=None, inv_y=True, stage='train', return_against=False):
    
    x_indice = None
    if against_hdf5 is not None:
        # extract the gene names 
        with h5py.File(input_hdf5, 'r') as f:
            genes_target = f['columns_x'][...].astype(str)
        with h5py.File(against_hdf5, 'r') as f:
            genes_against = f['columns_x'][...].astype(str)
        # get the genes occur in both models
        x_indice_target, x_indice_against = util_misc.intersect_indice(genes_target, genes_against) 
        x_indice = x_indice_target
                       
    
    feature_dic = util_hdf5.read_yaml(data_scheme_yaml)
    with h5py.File(input_hdf5, 'r') as f:
        features = f['columns_y'][:].astype('str')
        sample_size = f['y'].shape[0]
        y = f['y'][:]
    covar_indice = np.where(np.isin(features, feature_dic['covar_names']))[0]
    trait_indice = np.where(np.isin(features, feature_dic['outcome_names']))[0]
    
    logging.info('Features in order')
    logging.info(features)
    
    # load data_scheme for training
    batch_size_to_load = batch_size
    logging.info(f'batch_size in training set is {batch_size_to_load}')
    
    data_scheme, sample_size = util_hdf5.build_data_scheme(
        input_hdf5, 
        data_scheme_yaml, 
        batch_size=batch_size_to_load, 
        inv_norm_y=inv_y,
        x_indice=x_indice
    )
    
    if stage == 'train':
        # set validation and test set as the first and second batch
        # dataset_valid = data_scheme.dataset.take(1)
        data_scheme.dataset = data_scheme.dataset.skip(1)
        # dataset_test = data_scheme.dataset.take(1)
        data_scheme.dataset = data_scheme.dataset.skip(1)
        batch_size = sample_size // 8 + 1
        data_scheme.dataset = data_scheme.dataset.unbatch().batch(batch_size)
        
        # dataset_insample = data_scheme.dataset.take(1)
        ntrain = sample_size - batch_size_to_load * 2
        train_batch = batch_size
        logging.info(f'train_batch = {train_batch}, ntrain = {ntrain}')
        
        return data_scheme, ntrain, train_batch
        
    
    elif stage == 'test':
        _, dataset_valid, dataset_test, dataset_insample = split_dataset_into_test_and_valid(data_scheme)
        
        if return_against is False or against_hdf5 is None:
            return dataset_valid, dataset_test, dataset_insample, (features, trait_indice), x_indice
        else:
            data_scheme_against, sample_size_against = util_hdf5.build_data_scheme(
                against_hdf5, 
                data_scheme_yaml, 
                batch_size=batch_size_to_load, 
                inv_norm_y=inv_y,
                x_indice=x_indice_against
            )
            _, dataset_valid_aga, dataset_test_aga, dataset_insample_aga = split_dataset_into_test_and_valid(data_scheme_against)
            return dataset_valid, dataset_test, dataset_insample, (features, trait_indice), (dataset_valid_aga, dataset_test_aga, dataset_insample_aga, x_indice_target, x_indice_against)
    
    elif stage == 'export':
        gene_list = genes_target[data_scheme.get_indice_x()]
        trait_list = features[data_scheme.outcome_indice]
        covar_list = features[data_scheme.covariate_indice]
        return gene_list, trait_list, covar_list
        
def split_dataset_into_test_and_valid(data_scheme):
    dataset_valid = data_scheme.dataset.take(1)
    data_scheme.dataset = data_scheme.dataset.skip(1)
    dataset_test = data_scheme.dataset.take(1)
    data_scheme.dataset = data_scheme.dataset.skip(1)
    dataset_insample = data_scheme.dataset.take(1)
    return data_scheme, dataset_valid, dataset_test, dataset_insample


def save_list(mylist, output):
    with open(output, 'w') as f:
        for l in mylist:
            f.write(l + '\n')

def gen_dir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        print("Directory " , dirname ,  " Created ")
    else:    
        print("Directory " , dirname ,  " already exists")
