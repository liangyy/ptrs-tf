#!/usr/bin/env python
# coding: utf-8

# In[1]:


# logfile = '/vol/bmd/yanyul/UKB/ptrs-tf/models/elastic_net.log'
output_dir = '/vol/bmd/yanyul/UKB/ptrs-tf/models'
population = 'Chinese'  # 'British'  # for test 
batch_size = 512  # 2 ** 12
logfile = f'{output_dir}/cnn_{population}_ctimp_Whole_Blood.log'


# In[2]:


import sys, re
sys.path.append("../code/")
import lib_LinearAlgebra, lib_cnnPTRS, util_hdf5, util_misc
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py, yaml
import matplotlib.pyplot as plt
from importlib import reload  
lib_LinearAlgebra = reload(lib_LinearAlgebra)
lib_cnnPTRS = reload(lib_cnnPTRS)
util_hdf5 = reload(util_hdf5)
util_misc = reload(util_misc)
import logging, sys
import seaborn as sns
logging.basicConfig(
    level = logging.INFO, 
#     stream = sys.stderr,
    filename = logfile,
    format = '%(asctime)s  %(message)s',
    datefmt = '%Y-%m-%d %I:%M:%S %p'
)


# # Analysis overview
# 
# Building PTRS using `cnnPTRS`. 
# 
# 1. Split British data into 3 sets: training, test, validation.
# 2. Train a cnnPTRS model using British training data. First train everything.
# 3. Fixing covariate weights, train CNN only.
# 

# # Load spatial information

# In[ ]:


def get_tss(start, end, strand):
    if strand == '+':
        return start
    else:
        return end
def chr2num(chrm):
    if 'X' in chrm:
        chrm = 23
    elif 'Y' in chrm:
        chrm = 24
    elif 'M' in chrm:
        chrm = 25
    else:
        chrm = int(re.sub('chr', '', chrm))
    return chrm

df_gene = pd.read_table('https://bitbucket.org/yanyul/rotation-at-imlab/raw/85a3fbe8f08df7c67265fed69569b7ea554d4e12/data/annotations_gencode_v26.tsv')


df_gene['tss'] = df_gene[['start', 'end', 'strand']].apply(lambda x: get_tss(x.start, x.end, x.strand), axis = 1)

df_gene['chr_num'] = df_gene[['chromosome']].apply(lambda x: chr2num(x.chromosome), axis = 1)

df_gene.sort_values(['chr_num', 'tss'], ascending = [True, True], inplace = True) 

df_gene = df_gene.reset_index(drop = True)

df_gene['rank'] = df_gene.index


# In[ ]:


with h5py.File(f'/vol/bmd/yanyul/UKB/predicted_expression_tf2/ukb_imp_x_ctimp_Whole_Blood_{population}.hdf5', 'r') as f:
    col_genes = f['columns_x'][...]
col_genes_cleaned = [ i.astype(str).split('.')[0] for i in col_genes ]
df_col_genes = pd.DataFrame({'gene_id': col_genes_cleaned, 'col_idx': [ i for i in range(len(col_genes_cleaned)) ]})

df_gene_joined = df_gene.join(df_col_genes.set_index('gene_id'), on = 'gene_id')

df_gene_joined = df_gene_joined.loc[df_gene_joined['gene_id'].isin(df_col_genes['gene_id'].to_list())].reset_index(drop = True)


x_indice = [ int(i) for i in df_gene_joined['col_idx'].to_list() ]

logging.info('Getting spatial data finished')


# # Load data

# In[3]:


logging.info(f'Loading data: {population}')
# set path to British data
hdf5_british = f'/vol/bmd/yanyul/UKB/predicted_expression_tf2/ukb_imp_x_ctimp_Whole_Blood_{population}.hdf5'

# data scheme specifying which are traits and covariates
scheme_yaml = '../misc_files/data_scheme.yaml'

# loading names of traits/covariates
# the order is matched with the data being loaded
feature_dic = util_hdf5.read_yaml(scheme_yaml)
with h5py.File(hdf5_british, 'r') as f:
    features = f['columns_y'][:].astype('str')
    sample_size = f['y'].shape[0]
    y = f['y'][:]
covar_indice = np.where(np.isin(features, feature_dic['covar_names']))[0]
trait_indice = np.where(np.isin(features, feature_dic['outcome_names']))[0]


# In[6]:


# load data_scheme for training
batch_size_to_load = batch_size  # int(sample_size / 8) + 1
print(f'batch_size in {population} set is {batch_size_to_load}', file = sys.stderr)
data_scheme, sample_size = util_hdf5.build_data_scheme(
    hdf5_british, 
    scheme_yaml, 
    batch_size = batch_size, 
    inv_norm_y = True
)

# set validation and test set as the first and second batch
dataset_valid = data_scheme.dataset.take(1)
data_scheme.dataset = data_scheme.dataset.skip(1)
# dataset_test = data_scheme.dataset.take(1)
data_scheme.dataset = data_scheme.dataset.skip(1)
dataset_insample = data_scheme.dataset.take(1)


# Prepare validation, insample, and test tensors.

# In[ ]:


logging.info('Preparing tensors')
# ele_test = util_misc.get_inputs_and_y(dataset_test, data_scheme.get_num_outcome())
ele_insample = util_misc.get_inputs_and_y(dataset_insample, data_scheme.get_num_outcome())
ele_valid = util_misc.get_inputs_and_y(dataset_valid, data_scheme.get_num_outcome())


# # Training

# In[7]:


logging.info('Start training')
cnn_model = util_misc.load_ordered_yaml('../misc_files/cnn_ptrs.yaml')
cnn = lib_cnnPTRS.cnnPTRS(cnn_model, data_scheme, f'{output_dir}/cnnPTRS_phase1_{population}_ctimp_Whole_Blood.h5', normalizer = True)
cnn.model.summary()


# Train all (phase1).

# In[ ]:


logging.info('Start to initialize normalizer')
optimizer = tf.keras.optimizers.Adam()
norm, norm_v = cnn.prep_train(ele_valid) 
logging.info('Normalizer initialization finished')


# In[ ]:


logging.info('Creating training graph: phase 1')
mytrain = cnn.train_func()
logging.info('Start training: Phase 1')
mytrain(cnn, optimizer, 20, ele_valid, normalizer = norm, normalizer_valid = norm_v, ele_insample = ele_insample)  # , log_path = 'file://' + logfile
logging.info('Training Phase 1 finished')


# Train CNN only (phase 2).

# In[ ]:


cnn.temp_path = f'{output_dir}/cnnPTRS_phase2_{population}_ctimp_Whole_Blood.h5'
var_list = cnn.model.trainable_variables
_ = var_list.pop(-1)
_ = var_list.pop(-1)
var_list[-1]
logging.info('Creating training graph: phase 2')
mytrain = cnn.train_func(var_list = var_list)
logging.info('Start training: Phase 2')
mytrain(cnn, optimizer, 2000, ele_valid, normalizer = norm, normalizer_valid = norm_v, ele_insample = ele_insample)  # , log_path = 'file://' + logfile)
logging.info('Training Phase 2 finished')


# Save model.

# In[ ]:


logging.info('Saving model')
cnn.model.save(f'{output_dir}/cnnPTRS_{population}_ctimp_Whole_Blood.h5')

