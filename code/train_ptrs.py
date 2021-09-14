import numpy as np
import pandas as pd
import util_Stats
from util_Stats import calc_auc
from util_misc import load_ordered_yaml

def parse_data_args(args):
    return args.split(':')

def _pr2_format(ele, features, name, alpha, lambda_):
    nlambda = lambda_.shape[1]
    ntrait = lambda_.shape[0]
    ele_seq = np.reshape(ele, (nlambda * ntrait), order = 'C')
    lambda_seq = np.reshape(lambda_, (nlambda * ntrait), order = 'C')
    f_seq = np.repeat(features, nlambda)
    return pd.DataFrame({'partial_r2': ele_seq, 'trait': f_seq, 'sample': name, 'alpha': alpha, 'lambda': lambda_seq})

def predict_only(alpha_list, model_list, dataset_dict, dataset_eid_dict, features, simple=False):
    res = []
    for alpha in alpha_list:
        model_i = model_list[alpha]
        for i in dataset_dict.keys():
            if i not in dataset_eid_dict:
                continue
            dataset = dataset_dict[i]
            if simple is False:
                for ele in dataset:
                    x, y = model_i.data_scheme.get_data_matrix(ele)
                    covar = x[:, -len(model_i.data_scheme.covariate_indice) :]
                    print('alpha = {}, trait = {}, ncol(covar) = {}'.format(alpha, i, covar.shape[1]))
                out = model_i.predict_x(dataset, model_i.beta_hat_path)
            else:
                out = {}
                covar, out['y'], out['y_pred_from_x'] = dataset
            tmp_flat = []
            for fidx, feature in enumerate(features):
                tmp = pd.DataFrame(
                    out['y_pred_from_x'][:, fidx, :], 
                    columns=[ f'alpha_{alpha}_feature_{feature}_hyper_param_{j}' for j in range(out['y_pred_from_x'].shape[2]) ])
                tmp_flat.append(tmp)
            tmp_flat = pd.concat(tmp_flat, axis=1)
            tmp_flat['eid'] = dataset_eid_dict[i]
            res.append(tmp_flat)    
    res = pd.concat(res, axis=0)
    return res   

def get_partial_r2(alpha_list, model_list, dataset_dict, features, binary=False, split_yaml=None, simple=False):
    if split_yaml is None:
        syaml = None
    else:
        syaml = load_ordered_yaml(split_yaml)
        if 'nrepeat' not in syaml:
            syaml['nrepeat'] = 10
        if 'fraction' not in syaml:
            syaml['fraction'] = 0.5
        if 'seed' not in syaml:
            syaml['seed'] = 1
        np.random.seed(syaml['seed'])
    partial_r2 = {}
    for alpha in alpha_list:
        partial_r2[alpha] = {}
        model_i = model_list[alpha]
        for i in dataset_dict.keys():
            dataset = dataset_dict[i]
            if simple is False:
                for ele in dataset:
                    x, y = model_i.data_scheme.get_data_matrix(ele)
                    covar = x[:, -len(model_i.data_scheme.covariate_indice) :]
                    print('alpha = {}, trait = {}, ncol(covar) = {}'.format(alpha, i, covar.shape[1]))
                out = model_i.predict_x(dataset, model_i.beta_hat_path)
            else:
                out = {}
                covar, out['y'], out['y_pred_from_x'] = dataset
            if syaml is None:
                if binary is False:
                    partial_r2[alpha][i] = util_Stats.quick_partial_r2(covar, out['y'], out['y_pred_from_x'])
                else:
                    partial_r2[alpha][i] = util_Stats.binary_perf(covar, out['y'], out['y_pred_from_x'], func=calc_auc)
            else:
                out2 = []
                labels = []
                ntotal = out['y'].shape[0]
                nselect = int(ntotal * syaml['fraction'])
                idx_all = np.arange(ntotal)
                for ii in range(syaml['nrepeat']):
                    selected_idx = np.random.choice(ntotal, nselect, replace=False)
                    selected_ind = np.isin(idx_all, selected_idx)
                    yy1 = out['y'][selected_ind]
                    yy2 = out['y'][~selected_ind]
                    yyp1 = out['y_pred_from_x'][selected_ind, :]
                    yyp2 = out['y_pred_from_x'][~selected_ind, :]
                    if not isinstance(covar, np.ndarray):
                        cc = covar.numpy()
                    else:
                        cc = covar.copy()
                    cc1 = cc[selected_ind, :]
                    cc2 = cc[~selected_ind, :]
                    if binary is False:
                        tmp1 = util_Stats.quick_partial_r2(cc1, yy1, yyp1)
                        tmp2 = util_Stats.quick_partial_r2(cc2, yy2, yyp2)
                    else:
                        tmp1 = util_Stats.binary_perf(cc1, yy1, yyp1, func=calc_auc)
                        tmp2 = util_Stats.binary_perf(cc2, yy2, yyp2, func=calc_auc)
                    out2.append(tmp1)
                    out2.append(tmp2)
                    labels.append(f'repeat{ii}_1')
                    labels.append(f'repeat{ii}_2')
                partial_r2[alpha][i] = (out2, labels)
                    
    res_list = []
    if syaml is None:
        df = pd.DataFrame({'partial_r2': [], 'trait': [], 'sample': [], 'alpha': [], 'lambda': []})
    else:
        df = pd.DataFrame({'partial_r2': [], 'trait': [], 'sample': [], 'alpha': [], 'lambda': [], 'split_label': []})
    for alpha in alpha_list:
        model_i = model_list[alpha]
        if simple is False:
            lambda_i = np.array(model_i.lambda_seq)
        else:
            lambda_i = np.array(model_i)
        for i in partial_r2[alpha].keys():
            if syaml is None:
                df = pd.concat((df, _pr2_format(partial_r2[alpha][i], features, i, alpha, lambda_i)))
            else:
                res = partial_r2[alpha][i]
                for oo, ll in zip(res[0], res[1]):
                    tmp_df1 = _pr2_format(oo, features, i, alpha, lambda_i)
                    # tmp_df2 = _pr2_format(oo2, features[trait_indice], i, alpha, lambda_i)
                    tmp_df1['split_label'] = ll
                    # tmp_df2['split_label'] = ll2
                    df = pd.concat((df, tmp_df1))
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
    parser.add_argument('--all_training', action='store_true', help='''
        If set, all data will be used for training. --size_of_data_to_hold won't be effective.
    ''')
    parser.add_argument('--prediction_only', action='store_true', help='''
        Do prediction if specified.
    ''')
    parser.add_argument('--data_hdf5', help='''
        Data in HDF5. 
        Use the format: NAME:PATH
    ''')
    parser.add_argument('--data_scheme_yaml', help='''
        Data scheme YAML.
    ''')
    parser.add_argument('--alpha_seq', nargs='+', type=float, default=[0.1, 0.5, 0.9], help='''
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
    parser.add_argument('--export', action='store_true', help='''
        If specified, it will export the --prediction_model into TXT format.
    ''')
    parser.add_argument('--lambda_dict', default=None, help='''
        If want to use another definition of lambda sequence, specify it here.
    ''')
    parser.add_argument('--pt_cutoffs', default=None, help='''
        This option is effective only in training mode. 
        If specified, it will run P+T mode instead. 
        The p-value cutoffs should be ","-delimited.
    ''')
    parser.add_argument('--split_yaml', default=None, help='''
        If set, it will split the test set accordingly and calculate the R2 for each split.
        The YAML should contain:
        nrepeat (default=10), fraction (default=0.5), seed (default=1)
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
    
    from train_lib import prep_dataset_from_hdf5, save_list, gen_dir
    import util_ElasticNet, lib_LinearAlgebra, util_hdf5, lib_ElasticNet, lib_Checker
    import tensorflow as tf
    import functools
    import scipy.stats
    # from util_misc import load_ordered_yaml


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
            against_hdf5=against_hdf5, inv_y=inv_y,
            all_training=args.all_training
        )
    else:
        if args.export is False:
            d_valid, d_test, d_insample, feature_tuple, more_info = prep_dataset_from_hdf5(
                data_hdf5, args.data_scheme_yaml, args.size_of_data_to_hold, logging, 
                against_hdf5=against_hdf5, inv_y=inv_y, return_against=True,
                stage='test',
                all_training=args.all_training
            )
            features, trait_indice = feature_tuple
            if args.against_hdf5 is not None:
                d_valid_aga, d_test_aga, d_insample_aga, x_indice, x_indice_aga = more_info
            else:
                x_indice = more_info
            model_list = {}
            for alpha in alpha_list:
                filename = args.prediction_model.format(alpha=alpha)
                model_list[alpha] = lib_LinearAlgebra.ElasticNetEstimator('', None, minimal_load=True)
                model_list[alpha].minimal_load(filename)
            
            if args.all_training is False:
                dataset_dict = {
                    f'{data_name}_valid': d_valid,
                    f'{data_name}_test': d_test,
                    f'{data_name}_insample': d_insample
                }
            else:
                dataset_dict = { f'{data_name}_insample': d_insample }
            dataset_eid_dict = {}
            if args.data_hdf5_predict is not None:
                batch_size_here = 8096
                for data_pred in args.data_hdf5_predict:
                    data_pred_name, data_pred_hdf5 = parse_data_args(data_pred)
                    data_scheme, _, data_eid = util_hdf5.build_data_scheme(
                        data_pred_hdf5, 
                        args.data_scheme_yaml, 
                        batch_size=batch_size_here, 
                        inv_norm_y=inv_y,
                        x_indice=x_indice, 
                        return_eid=True
                    )
                    dataset_dict[data_pred_name] = data_scheme.dataset
                    dataset_eid_dict[data_pred_name] = data_eid
            if args.against_hdf5 is not None:
                if args.all_training is False:
                    dataset_aga_dict = {
                        f'{against_name}_valid': d_valid_aga,
                        f'{against_name}_test': d_test_aga,
                        f'{against_name}_insample': d_insample_aga
                    }
                else:
                    dataset_dict = { f'{data_name}_insample': d_insample }
                dataset_aga_eid_dict = {}
                if args.against_hdf5_predict is not None:
                    batch_size_here = 8096
                    for against_pred in args.against_hdf5_predict:
                        against_pred_name, against_pred_hdf5 = parse_data_args(against_pred)
                        data_scheme, _, data_eid = util_hdf5.build_data_scheme(
                            against_pred_hdf5, 
                            args.data_scheme_yaml, 
                            batch_size=batch_size_here, 
                            inv_norm_y=inv_y,
                            x_indice=x_indice_aga, 
                            return_eid=True
                        )
                        dataset_aga_dict[against_pred_name] = data_scheme.dataset
                        dataset_aga_eid_dict[data_pred_name] = data_eid
        else:
            gene_list, trait_list, covar_list = prep_dataset_from_hdf5(
                data_hdf5, args.data_scheme_yaml, args.size_of_data_to_hold, logging, 
                against_hdf5=against_hdf5, inv_y=inv_y,
                stage='export',
                all_training=args.all_training
            )
            
    
    if args.prediction_model is None:
        ### Training
        learning_rate = 1
        out_prefix = args.out_prefix
        
        if args.pt_cutoffs is not None:
            z_cutoffs = [ scipy.stats.norm.isf(float(i)) for i in args.pt_cutoffs.split(',') ]
            
        for alpha in alpha_list:
            logging.info('alpha = {} starts'.format(alpha))
            if args.lambda_dict is None:
                lambda_init_dict = {
                    'data_init': None, 
                    'prefactor_of_lambda_max': 1.5,
                    'lambda_max_over_lambda_min': 1e6,
                    'nlambda': 50
                }
            else:
                lambda_init_dict = load_ordered_yaml(args.lambda_dict)
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
            if args.pt_cutoffs is None:
                checker = [ lib_Checker.Checker(ntrain, train_batch, lib_Checker.my_stat_fun, my_stop_rule) 
                           for i in range(ny) ]

                elastic_net_estimator.solve(checker, nepoch=100, logging=logging)
            else:
                elastic_net_estimator.solve_pt(abs_z_cutoffs=z_cutoffs)
            
            outfile = f'{out_prefix}_{alpha}.hdf5'
            logging.info(f'alpha = {alpha} saving to {outfile}')
            elastic_net_estimator.minimal_save(outfile)
            logging.info('alpha = {} ends'.format(alpha))
    else:
        if args.export is False:
            if args.prediction_only is False:
                ### Predict and get partial r2
                ### Do data_hdf5 first and then do against_hdf5 if needed
                res_list = []
                df = get_partial_r2(alpha_list, model_list, dataset_dict, features[trait_indice], binary=args.binary, split_yaml=args.split_yaml)
                df['pred_expr_source'] = 'train'
                res_list.append(df)
                
                ### Then do against_hdf5
                if args.against_hdf5 is not None:
                    # we need to first change the order of data to be loaded to match the against. 
                    for alpha in alpha_list:
                        model_list[alpha].data_scheme.x_indice = x_indice_aga
                    
                    df = get_partial_r2(
                        alpha_list, model_list, dataset_aga_dict, features[trait_indice],
                        binary=args.binary, split_yaml=args.split_yaml
                    )
                    df['pred_expr_source'] = 'against'
                    res_list.append(df)
                
                res = pd.concat(res_list, axis=0)
                
                res.to_csv(args.out_prefix + '.performance.csv', index=False)
            else:
                res_list = []
                df = predict_only(
                    alpha_list, model_list, dataset_dict,
                    dataset_eid_dict, features[trait_indice])
                df['pred_expr_source'] = 'train'
                res_list.append(df)
                if args.against_hdf5 is not None:
                    for alpha in alpha_list:
                        model_list[alpha].data_scheme.x_indice = x_indice_aga
                        df = predict_only(
                            alpha_list, model_list, dataset_aga_dict, features[trait_indice])
                        df['pred_expr_source'] = 'against'
                        res_list.append(df)
                res = pd.concat(res_list, axis=0)
                res.to_csv(args.out_prefix + '.prediction.csv.gz', index=False, compression='gzip')
        else:
            model_list = {}
            for alpha in alpha_list:
                filename = args.prediction_model.format(alpha=alpha)
                model_list[alpha] = lib_LinearAlgebra.ElasticNetEstimator('', None, minimal_load=True)
                model_list[alpha].minimal_load(filename)
            # save gene list, trait list, and covariate list
            for alpha in alpha_list:
                outfile_prefix = '{}_{}'.format(args.out_prefix, alpha)
                gene_out = outfile_prefix + '.gene_list.txt'
                save_list(gene_list, gene_out)
                trait_out = outfile_prefix + '.trait_list.txt'
                save_list(trait_list, trait_out)
                covar_out = outfile_prefix + '.covar_list.txt'
                save_list(covar_list, covar_out)
                outdir = outfile_prefix + '.export_model/'
                gen_dir(outdir)
                betas = model_list[alpha].beta_hat_path[:]
                gene_df = pd.DataFrame({'gene_id': gene_list})
                for tidx, trait in enumerate(trait_list):
                    print(f' Working on {trait}')
                    outputfile = outdir + f'weights.{trait}.tsv.gz'
                    weight_mat = betas[:, tidx, :].numpy()
                    weight_mat = weight_mat[:, np.abs(weight_mat).sum(axis=0) != 0]
                    weight_df = pd.concat((gene_df, pd.DataFrame(weight_mat, columns=[ f'model_{idx}' for idx in range(weight_mat.shape[1]) ])), axis=1)
                    weight_df.to_csv(outputfile, index=False, compression='gzip', sep='\t')
                    
                        
    
    
