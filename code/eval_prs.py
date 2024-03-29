import pandas as pd
import numpy as np
import h5py
import util_hdf5, util_Stats
from util_misc import load_ordered_yaml

def check_eq2(ll1, ll2):
    ee = np.array(ll1) == np.array(ll2)
    if np.sum(~ee) > 0:
        raise ValueError('ll1 and ll2 are not equal.')

def check_eq(l1, l2):
    if len(l1) != len(l2):
        raise ValueError('l1 and l2 have different number of elements.')
    for i, j in zip(l1, l2):
        if i != j:
            raise ValueError('l1 and l2 elements are in different order.')

def load_indiv(fn):
    reference_mat = pd.read_csv(fn, sep=' ')
    reference_mat = reference_mat.rename(columns = {'FID': 'indiv'})
    reference_mat = reference_mat.drop(['IID'], axis = 1)
    reference_mat['indiv'] = reference_mat['indiv'].astype('str')
    return reference_mat

def load_y_and_covar(pred_expr, reference_mat, args):
    with h5py.File(pred_expr, 'r') as f:
        features = f['columns_y'][...].astype('str')
        indiv_ids = f['rows'][...].astype('str')
    data_scheme, _ = util_hdf5.build_data_scheme(
        pred_expr, 
        args.data_scheme_yaml, 
        batch_size=args.size_of_data_to_hold, 
        inv_norm_y=inv_y
    )
    
    ymat = np.empty((0, data_scheme.get_num_covariate() + data_scheme.get_num_outcome()))
    for _, y in data_scheme.dataset:
        ymat = np.concatenate((ymat, y), axis = 0)
    g = pd.DataFrame(ymat)
    g['indiv'] = indiv_ids
    data_mat = reference_mat.join(g.set_index('indiv'), on = 'indiv').iloc[:, reference_mat.shape[1]:]
    ymat = data_mat.to_numpy()

    ## here
    covar = ymat[:, data_scheme.covariate_indice]
    y = ymat[:, data_scheme.outcome_indice]
    pheno_list = features[data_scheme.outcome_indice]
    
    return y, covar, pheno_list
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='eval_prs.py', description='''
        Evaluation PRS performance.
    ''')
    parser.add_argument('--logfile', help='''
        Log file path.
    ''')
    parser.add_argument('--data_scheme_yaml', help='''
        Data scheme YAML.
    ''')
    parser.add_argument('--data_hdf5_predict', default=None, nargs='+', help='''
        Specify the list of HDF5 data to predict on. 
        Here the data scheme should be the same as the data_hdf5.
        Use the format: NAME:PATH:INDIVLIST.
    ''')
    parser.add_argument('--prs_table', default=None, help='''
        Path to PRS table.
    ''')
    parser.add_argument('--ptrs_table', default=None, help='''
        Path to PTRS table.
    ''')
    parser.add_argument('--binary', action='store_true', help='''
        If specified, it will treat the yobs as binary values.
        And do partial R2 calculation based on logistic regression.
    ''')
    parser.add_argument('--split_yaml', default=None, help='''
        If set, it will split the test set accordingly and calculate the R2 for each split.
        The YAML should contain:
        nrepeat (default=10), fraction (default=0.5), seed (default=1)
    ''')
    parser.add_argument('--out_prefix', help='''
        Output prefix.
    ''')
    parser.add_argument('--no_inv_y', action='store_true', help='''
        If specified, will not apply inverse normalization to y.
    ''')
    parser.add_argument('--size_of_data_to_hold', type=int, help='''
        Two batches (the first two) of data will be held out for training.
        Specify the size of the batch here.
    ''')
    parser.add_argument('--prs_col_pattern', help='''
        PRS column name pattern (contain {trait} as a wildcard).
    ''')
    parser.add_argument('--ptrs_col_pattern', help='''
        PTRS column name pattern (contain {trait} as a wildcard).
    ''')
    args = parser.parse_args()
 
    import logging, time, sys, os, re
    # configing util
    logging.basicConfig(
        level=logging.INFO, 
        filename=args.logfile, 
        format='%(asctime)s  %(message)s',
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )
    from train_ptrs import parse_data_args, get_partial_r2
    # main body
   
    inv_y = not args.no_inv_y
    
    logging.info('Loading PRS table.')
    df_prs = pd.read_csv(args.prs_table, sep='\t', compression='gzip')
    df_prs['s'] = df_prs['s'].astype('str')
    ntotal = df_prs.shape[1] - 1
    
    if args.ptrs_table is not None:
        logging.info('Loading PTRS table.')
        df_ptrs = pd.read_csv(args.ptrs_table, compression='gzip')
        df_ptrs['eid'] = df_ptrs['eid'].astype('str')
        df_ptrs_new = pd.DataFrame({'eid': df_prs.s})
        df_ptrs = pd.merge(df_ptrs_new, df_ptrs, on='eid')
        ntotal_ptrs = df_ptrs.shape[1] - 1
    
    logging.info('Collecting y, yp, and covar by population.')
    collect_dic = {}
    # some placeholders to run get_partial_r2
    alpha_list = [ 'NA' ]
    model_list = { 'NA': None }
    dataset_dict = {}
    dataset_dict_2 = {}
    pheno_list = None
    for data_pred in args.data_hdf5_predict:
        data_pred_name, pred_expr, indiv_list = parse_data_args(data_pred)
        logging.info(f'Working on {data_pred_name}')
        reference_mat = load_indiv(indiv_list)
        y, covar, pheno_list2 = load_y_and_covar(pred_expr, reference_mat, args)
        if pheno_list is None:
            pheno_list = pheno_list2
        else:
            check_eq(pheno_list, pheno_list2)
        
        npheno = y.shape[1]
        npoints = ntotal // npheno
        prs_collector = np.empty((reference_mat.shape[0], npheno, npoints))
        hypers = []
        if args.ptrs_table is not None:
            npoints_ptrs = ntotal_ptrs // npheno
            ptrs_collector = np.empty((reference_mat.shape[0], npheno, npoints_ptrs))
            model_list = { 'NA': None }
        for i in range(len(pheno_list)):
            trait = pheno_list[i]
            cols = []
            colname_j = '^' + args.prs_col_pattern.format(trait=trait)
            for j in list(df_prs.columns):
                jj = '^' + j
                if colname_j in jj:
                    cols.append(j)
            prs_mat = df_prs[['s'] + cols]
            out_mat = reference_mat.join(prs_mat.set_index('s'), on='indiv').iloc[:, 1:]
            out_mat = out_mat.to_numpy()
            prs_collector[:, i, :] = out_mat   
            prs_names = [ re.sub(colname_j, '', i) for i in cols ]
            if args.ptrs_table is not None:
                cols_ptrs = []
                colname_ptrs_j = '^' + args.ptrs_col_pattern.format(trait=trait)
                for j in list(df_ptrs.columns):
                    jj = '^' + j
                    if colname_ptrs_j in jj:
                        cols_ptrs.append(j)
                ptrs_mat = df_ptrs[['eid'] + cols_ptrs]
                out_mat = reference_mat.join(ptrs_mat.set_index('eid'), on='indiv').iloc[:, 1:]
                out_mat = out_mat.to_numpy()
                ptrs_collector[:, i, :] = out_mat 
                ptrs_names = [ re.sub(colname_ptrs_j, '', i) for i in cols_ptrs ]
                combine_names = []
                for h1 in prs_names:
                    for h2 in ptrs_names:
                        combine_names.append(f'{h1}_x_{h2}')
                hypers.append(combine_names)   
            else:
                hypers.append(prs_names)
        if model_list['NA'] is None:
            model_list['NA'] = hypers
        else:
            check_eq2(model_list['NA'], hypers)
        dataset_dict[data_pred_name] = (covar, y, prs_collector)
        if args.ptrs_table is None:
            dataset_dict_2 = None
        else:
            dataset_dict_2[data_pred_name] = (None, None, ptrs_collector)
        
    logging.info('Calculating partial r2.')
    df = get_partial_r2(alpha_list, model_list, dataset_dict, pheno_list, binary=args.binary, split_yaml=args.split_yaml, simple=True, dataset_dict_2=dataset_dict_2)      
    
    df.to_csv(args.out_prefix + '.performance.csv', index=False)
    
