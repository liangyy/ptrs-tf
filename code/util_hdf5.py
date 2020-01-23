import numpy as np
import pandas as pd
import yaml
import h5py

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
        
