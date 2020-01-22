import argparse
parser = argparse.ArgumentParser(prog='split_pred_expr.py', description='''
    Split predicted expression in HDF5 format into smaller chunks (to enable
    efficient I/O in TF2). 
''')

parser.add_argument('--output-prefix', required=True, help='''
    Phenotype table for subset individuals
''')
parser.add_argument('--yaml-of-inputs', required=True, help='''
    Three sections: 
        1. for pheno csv, named as 'pheno_csv'; 
        2. for indiv list, names as 'indiv_list'; 
        3. for pred expr hdf5, named as 'pred_expr_hdf5'.
    For 1, it specifies which column is sample ID and which columns should be 
    included in output. For example:
                        file_path: 'path_to_hdf5'
                        sample_col: 'name'
                        output_col:
                          - 'col1'
                          - 'col2'
    For 2, it specifies the path to lists. And also which column is 
    sample ID and separator of the file. For example:
                        lists:
                          pop1: 'file1'
                          pop2: 'file2'
                        sample_col: 'name'
                        sep: ' '
    For 3, it specifies dataset names for sample ID, gene ID, and the matrix.
    IMPORTANT: it is assuming gene in rows and sample in columns!
    For example:
                        file_path: 'path_to_csv'
                        dataset_sample: 'samples'
                        dataset_gene: 'genes'
                        pred_expr: 'pred_expr'
''')

args = parser.parse_args()

import pandas as pd
import numpy as np
import util_hdf5

# configing util
logging.basicConfig(
    level = logging.INFO, 
    stream = sys.stderr, 
    format = '%(asctime)s  %(message)s',
    datefmt = '%Y-%m-%d %I:%M:%S %p'
)

# load yaml
mydic = util_hdf5.read_yaml(args.yaml_of_inputs)

# load phenotype data
pheno = pd.read_csv(mydic['pheno_csv']['file_path'])
pheno_sample_col = mydic['pheno_csv']['sample_col']
## to make sure the sample ID column is string
pheno[pheno_sample_col] = pheno[pheno_sample_col].astype('str')

# load indiv lists
indiv_sample_col = mydic['indiv_list']['sample_col']
pop_dic = {}
for i in mydic['indiv_list']['lists'].keys():
    logging.info('Loading indiv list {}'.format(i))
    filename = mydic['indiv_list']['lists'][i]
    popname = i
    df_pop = pd.read_table('/vol/bmd/yanyul/GitHub/ptrs-ukb/output/data_split/African.txt', header = 0, sep = mydic['indiv_list']['sep'])
    ## to make sure the sample ID column is string
    df_pop[indiv_sample_col] = df_pop[indiv_sample_col].astype('str')
    ## annotate indiv df with phenotype features
    df_pop = df_pop.join(pheno.set_index(pheno_sample_col), on = indiv_sample_col)
    ## rename the sample ID column to sample
    df_pop = df_pop.rename(columns = {'IID': 'sample'})
    ## add it to pop_dic
    pop_dic[i] = df_pop

# do the splitting
f = split_hdf5_into_chunks(
    mydic['pred_expr_hdf5']['file_path'],
    mydic['pred_expr_hdf5']['dataset_sample'],
    mydic['pred_expr_hdf5']['dataset_gene'],
    mydic['pred_expr_hdf5']['pred_expr'],
    pop_dic,
    args.output_prefix,
    logging
)
logging.info('List of output files: {}'.format('\n'.join(f)))
