import numpy as np
import pandas as pd
import yaml
import h5py
import util_ElasticNet
import lib_LinearAlgebra

def update_cols(mat, new_submat, exclude_idx = None):
    if exclude_idx is None:
        return new_submat
    else:
        out = mat
        include_idx = []
        for i in range(out.shape[1]):
            if i not in exclude_idx:
                include_idx.append(i)
        # breakpoint()
        out[:, include_idx] = new_submat
        return out
def extract_cols(h5mat, col_names, target_names):
    col_names = np.array(col_names)
    target_names = np.array(target_names)
    target_idx = np.where(np.isin(col_names, target_names))[0].flatten().tolist()
    subcols = col_names[target_idx]
    submat = h5mat[:, target_idx]
    return submat, subcols
def split_hdf5_into_chunks(filename, dataset_sample, dataset_gene, dataset_mat, pop_dict, output_prefix, logging = None):
    with h5py.File(filename, 'r') as h5handle:
        sample_names = h5handle[dataset_sample][:].astype('str')
        mat = h5handle[dataset_mat]
        outfiles = []
        nlist = len(list(pop_dict.keys()))
        counter = 1
        for i in pop_dict.keys():
            if logging is not None:
                logging.info('split_hdf5_into_chunks: working on {}/{}: {}'.format(counter, nlist, i))
            df_indiv = pop_dict[i]
            sub_mat, sub_indiv = extract_cols(mat, sample_names, df_indiv['sample'].to_list())
            df_sub_mat = pd.DataFrame({'sample': sub_indiv})
            y = df_sub_mat.join(df_indiv.set_index('sample'), on = 'sample')
            y_sample = y['sample'].to_numpy()
#             print(y_sample)
            y_features = y.drop(['sample'], axis = 1)
            y_mat = y_features.to_numpy()
            # y = np.array(y)[:, np.newaxis]
            if len(y.shape) == 1:
                y = y[:, np.newaxis]
            outfile = '{}_{}.hdf5'.format(output_prefix, i)
            outfiles.append(outfile)
            with h5py.File(outfile, "w") as fx:
                fx.create_dataset("X", data = sub_mat.transpose())
                fx.create_dataset("y", data = y_mat)
                fx.create_dataset("rows", data = y_sample.astype('S'))
                fx.create_dataset("columns_x", data = h5handle[dataset_gene][:])
                fx.create_dataset("columns_y", data = y_features.columns.values.astype('S'))
            counter += 1
    return outfiles
def read_yaml(filename, gcp_project = None):
    with open(filename, 'r') as f:
        mydic = yaml.safe_load(f)        
    return mydic
def build_data_scheme(hdf5, scheme_yaml, batch_size = 128, num_epochs = 1, inv_norm_y = False, x_indice = None, return_eid = False):
    '''
    Assume HDF5 generated by `split_hdf5_into_chunks`
    '''
    mydic = read_yaml(scheme_yaml)
    # get column names in y
    with h5py.File(hdf5, 'r') as f:
        features = f['columns_y'][:].astype('str')
        eid = f['rows'][:].astype('str')
    # extract indice for covar and outcome
    covar_indice = np.where(np.isin(features, mydic['covar_names']))[0]
    outcome_indice = np.where(np.isin(features, mydic['outcome_names']))[0]
    # load dataset
    dataset, sample_size = util_ElasticNet.load_hdf5_as_dataset(
        [hdf5, hdf5],
        ['/X', '/y'],
        batch_size,
        num_epochs,
        inv_norm_y = inv_norm_y,
        covar_indice = covar_indice
    )
    tmp_x_indice = None
    if x_indice is not None:
        tmp_x_indice = list(x_indice)
    data_scheme = lib_LinearAlgebra.DataScheme(
        dataset = dataset, 
        X_index = mydic['X_index'],
        Y_index = mydic['Y_index'],
        outcome_indice = list(outcome_indice), 
        covariate_indice = list(covar_indice),
        x_indice = tmp_x_indice 
    )
    if return_eid is False:
        return data_scheme, sample_size
    else:
        return data_scheme, sample_size, eid
def build_data_scheme_with_preset_y(hdf5, scheme_yaml, preset_y, batch_size = 128, num_epochs = 1, x_indice = None):
    '''
    Assume HDF5 generated by `split_hdf5_into_chunks`
    `preset_y` should be tf.Tensor
    '''
    mydic = read_yaml(scheme_yaml)
    # get column names in y
    with h5py.File(hdf5, 'r') as f:
        features = f['columns_y'][:].astype('str')
    # extract indice for covar and outcome
    covar_indice = np.where(np.isin(features, mydic['covar_names']))[0]
    outcome_indice = np.where(np.isin(features, mydic['outcome_names']))[0]
    # load dataset
    dataset, sample_size = util_ElasticNet.load_hdf5_as_dataset(
        [hdf5, hdf5],
        ['/X', '/y'],
        batch_size,
        num_epochs,
        inv_norm_y = False,
        covar_indice = covar_indice,
        preset_y = preset_y
    )
    tmp_x_indice = None
    if x_indice is not None:
        tmp_x_indice = list(x_indice)
    data_scheme = lib_LinearAlgebra.DataScheme(
        dataset = dataset, 
        X_index = mydic['X_index'],
        Y_index = mydic['Y_index'],
        outcome_indice = list(outcome_indice), 
        covariate_indice = list(covar_indice),
        x_indice = tmp_x_indice 
    )
    return data_scheme, sample_size
        
