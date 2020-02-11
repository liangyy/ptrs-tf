import numpy as np
import yaml
import yamlloader

def load_ordered_yaml(filename):
    with open(filename) as yaml_file:
        data = yaml.load(yaml_file, Loader = yamlloader.ordereddict.CLoader)
    return data
                     
def intersect_indice(set1, set2):
    '''
    Return the indices of element in intersect of set1 and set2.
    The ordering of indices result in the same element order.
    Example:
        input:
            set1 = [1,3,4,5]
            set2 = [5,6,7,1]
        output:
            indice1 = [0,3]
            indice2 = [3,0]
    '''
    set_both = np.intersect1d(set1, set2)
    return _extract_subset_indice_with_sorting(set1, set_both), _extract_subset_indice_with_sorting(set2, set_both)
def _extract_subset_indice_with_sorting(extract_from, target_set):
    sorted_idx = np.argsort(extract_from)
    in_idx_in_sorted = np.where(np.isin(extract_from[sorted_idx], target_set))[0]
    sorted_idx_with_in = sorted_idx[in_idx_in_sorted]
    return sorted_idx_with_in
