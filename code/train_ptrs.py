

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
    inv_y = not args.no_inv_y
    data_scheme, ntrain, train_batch = prep_dataset_from_hdf5(
        args.data_hdf5, args.data_scheme_yaml, args.size_of_data_to_hold, logging, 
        against_hdf5=args.against_hdf5, inv_y=inv_y
    )
    
    ### Training
    alpha_list = args.alpha_seq
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
            
    
    
