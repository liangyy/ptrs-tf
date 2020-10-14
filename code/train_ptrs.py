import numpy as np
import pandas as pd
import util_Stats
from util_Stats import calc_auc

def parse_data_args(args):
    return args.split(':')

def _pr2_format(ele, features, name, alpha, lambda_):
    nlambda = lambda_.shape[1]
    ntrait = lambda_.shape[0]
    ele_seq = np.reshape(ele, (nlambda * ntrait), order = 'C')
    lambda_seq = np.reshape(lambda_, (nlambda * ntrait), order = 'C')
    f_seq = np.repeat(features, nlambda)
    return pd.DataFrame({'partial_r2': ele_seq, 'trait': f_seq, 'sample': name, 'alpha': alpha, 'lambda': lambda_seq})

def get_partial_r2(alpha_list, model_list, dataset_dict, binary=False):
    partial_r2 = {}
    for alpha in alpha_list:
        partial_r2[alpha] = {}
        model_i = model_list[alpha]
        for i in dataset_dict.keys():
            dataset = dataset_dict[i]
            for ele in dataset:
                x, y = model_i.data_scheme.get_data_matrix(ele)
                covar = x[:, -len(model_i.data_scheme.covariate_indice) :]
                print('alpha = {}, trait = {}, ncol(covar) = {}'.format(alpha, i, covar.shape[1]))
            out = model_i.predict_x(dataset, model_i.beta_hat_path)
            if binary is False:
                partial_r2[alpha][i] = util_Stats.quick_partial_r2(covar, out['y'], out['y_pred_from_x'])
            else:
                partial_r2[alpha][i] = util_Stats.binary_perf(covar, out['y'], out['y_pred_from_x'], func=calc_auc)
    res_list = []
    df = pd.DataFrame({'partial_r2': [], 'trait': [], 'sample': [], 'alpha': [], 'lambda': []})
    for alpha in alpha_list:
        model_i = model_list[alpha]
        lambda_i = np.array(model_i.lambda_seq)
        for i in partial_r2[alpha].keys():
            df = pd.concat((df, _pr2_format(partial_r2[alpha][i], features[trait_indice], i, alpha, lambda_i)))
    if binary is True:
        df.rename(columns={'partial_r2': 'roc_auc'}, inplace=True)
    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='train_ptrs.py', description='''
        Train PTRS model.
    ''')
    parser.add_argument('--logfile', help='''
        Log file path.
    ''')
    parser.add_argument('--out_prefix', help='''
        Directory of output.
    ''')
    parser.add_argument('--size_of_data_to_hold', type=int, help='''
        Two batches (the first two) of data will be held out for training.
        Specify the size of the batch here.
    ''')
    parser.add_argument('--data_hdf5', help='''
        Data in HDF5. 
        Use the format: NAME:PATH
    ''')
    parser.add_argument('--data_scheme_yaml', help='''
        Data scheme YAML.
    ''')
    parser.add_argument('--alpha_seq', nargs='+', type=int, default=[0.1, 0.5, 0.9], help='''
        A sequence of alpha to use.
    ''')
    parser.add_argument('--no_inv_y', action='store_true', help='''
        If specified, will not apply inverse normalization to y.
    ''')
    parser.add_argument('--against_hdf5', default=None, help='''
        Specify another HDF5 data where we will take the intersection of x as predictor.
        Use the format: NAME:PATH
    ''')
    parser.add_argument('--prediction_model', default=None, help='''
        Specify the prediction model. Use wildcard {alpha}.
        If it is specified, the script will switch to prediction mode.
        And use the model specified here to make the prediction.
    ''')
    parser.add_argument('--against_hdf5_predict', default=None, nargs='+', help='''
        Specify the list of HDF5 data to predict on. 
        Here the data scheme should be the same as the against_hdf5.
        Use the format: NAME:PATH.
    ''')
    parser.add_argument('--data_hdf5_predict', default=None, nargs='+', help='''
        Specify the list of HDF5 data to predict on. 
        Here the data scheme should be the same as the data_hdf5.
        Use the format: NAME:PATH.
    ''')
    parser.add_argument('--binary', action='store_true', help='''
        If specified, it will treat the yobs as binary values.
        And do partial R2 calculation based on logistic regression.
    ''')
    args = parser.parse_args()
 
    import logging, time, sys, os
    # configing util
    logging.basicConfig(
        level = logging.INFO, 
        filename = args.logfile,
        format = '%(asctime)s  %(message)s',
        datefmt = '%Y-%m-%d %I:%M:%S %p'
    )
    
    from train_lib import prep_dataset_from_hdf5
    import util_ElasticNet, lib_LinearAlgebra, util_hdf5, lib_ElasticNet, lib_Checker
    import tensorflow as tf
    import functools


    ### Load data
    alpha_list = args.alpha_seq
    inv_y = not args.no_inv_y
    data_name, data_hdf5 = parse_data_args(args.data_hdf5)
    if args.against_hdf5 is not None:
        against_name, against_hdf5 = parse_data_args(args.against_hdf5)
    else:
        against_hdf5 = None

    if args.prediction_model is None:
        data_scheme, ntrain, train_batch = prep_dataset_from_hdf5(
            data_hdf5, args.data_scheme_yaml, args.size_of_data_to_hold, logging, 
            against_hdf5=against_hdf5, inv_y=inv_y
        )
    else:

        d_valid, d_test, d_insample, feature_tuple, more_info = prep_dataset_from_hdf5(
            data_hdf5, args.data_scheme_yaml, args.size_of_data_to_hold, logging, 
            against_hdf5=against_hdf5, inv_y=inv_y, return_against=True,
            stage='test'
        )
        features, trait_indice = feature_tuple
        model_list = {}
        for alpha in alpha_list:
            filename = args.prediction_model.format(alpha=alpha)
            model_list[alpha] = lib_LinearAlgebra.ElasticNetEstimator('', None, minimal_load=True)
            model_list[alpha].minimal_load(filename)
        
        dataset_dict = {
            f'{data_name}_valid': d_valid,
            f'{data_name}_test': d_test,
            f'{data_name}_insample': d_insample
        }
        if args.data_hdf5_predict is not None:
            batch_size_here = 8096
            for data_pred in args.data_hdf5_predict:
                data_pred_name, data_pred_hdf5 = parse_data_args(data_pred)
                data_scheme, _ = util_hdf5.build_data_scheme(
                    data_pred_hdf5, 
                    args.data_scheme_yaml, 
                    batch_size=batch_size_here, 
                    inv_norm_y=inv_y,
                    x_indice=more_info
                )
                dataset_dict[data_pred_name] = data_scheme.dataset
        if args.against_hdf5 is not None:
            d_valid_aga, d_test_aga, d_insample_aga, x_indice, x_indice_aga = more_info
            dataset_aga_dict = {
                f'{against_name}_valid': d_valid_aga,
                f'{against_name}_test': d_test_aga,
                f'{against_name}_insample': d_insample_aga
            }
            if args.against_hdf5_predict is not None:
                batch_size_here = 8096
                for against_pred in args.against_hdf5_predict:
                    against_pred_name, against_pred_hdf5 = parse_data_args(against_pred)
                    data_scheme, _ = util_hdf5.build_data_scheme(
                        against_pred_hdf5, 
                        args.data_scheme_yaml, 
                        batch_size=batch_size_here, 
                        inv_norm_y=inv_y,
                        x_indice=x_indice_aga
                    )
                    dataset_aga_dict[data_pred_name] = data_scheme.dataset
    
    if args.prediction_model is None:
        ### Training
        learning_rate = 1
        out_prefix = args.out_prefix

        for alpha in alpha_list:
            logging.info('alpha = {} starts'.format(alpha))
            lambda_init_dict = {
                'data_init': None, 
                'prefactor_of_lambda_max': 1.5,
                'lambda_max_over_lambda_min': 1e6,
                'nlambda': 50
            }
            updater = lib_ElasticNet.ProximalUpdater(learning_rate=learning_rate, line_search=True)
            update_dic = {
                'updater': updater,
                'update_fun': updater.proximal_train_step
            }
            my_stop_rule = functools.partial(lib_Checker.diff_stop_rule, threshold=1e-3)
            ny = len(data_scheme.outcome_indice)
            elastic_net_estimator = lib_LinearAlgebra.ElasticNetEstimator(
                data_scheme,
                alpha,
                normalizer=True,
                learning_rate=learning_rate,
                lambda_init_dict=lambda_init_dict,
                updater=update_dic
            )
            checker = [ lib_Checker.Checker(ntrain, train_batch, lib_Checker.my_stat_fun, my_stop_rule) 
                       for i in range(ny) ]

            elastic_net_estimator.solve(checker, nepoch=100, logging = logging)
            
            outfile = f'{out_prefix}_{alpha}.hdf5'
            logging.info(f'alpha = {alpha} saving to {outfile}')
            elastic_net_estimator.minimal_save(outfile)
            logging.info('alpha = {} ends'.format(alpha))
    else:
        ### Predict and get partial r2
        ### Do data_hdf5 first and then do against_hdf5 if needed
        res_list = []
        df = get_partial_r2(alpha_list, model_list, dataset_dict, binary=args.binary)
        df['pred_expr_source'] = 'train'
        res_list.append(df)
        
        ### Then do against_hdf5
        if args.against_hdf5 is not None:
            # we need to first change the order of data to be loaded to match the against. 
            for alpha in alpha_list:
                model_list[alpha].data_scheme.x_indice = x_indice_aga
            
            df = get_partial_r2(alpha_list, model_list, dataset_aga_dict, binary=args.binary)
            df['pred_expr_source'] = 'against'
            res_list.append(df)
        
        res = pd.concat(res_list, axis=0)
        
        res.to_csv(args.out_prefix + '.perfermance.csv', index=False)
            
        
            
    
    
