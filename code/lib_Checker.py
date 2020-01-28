import numpy as np

def my_stat_fun(np_1dim_array):
    return np.mean(np_1dim_array)
def diff_stop_rule(summary, threshold):
    n = len(summary)
    if n >= 2:
        return (summary[n - 2] - summary[n - 1]) / max(abs(summary[n - 2]), abs(summary[n - 1])) < threshold
    else:
        return False
class Checker:
    def __init__(self, sample_size, batch_size, stat_fun, stop_rule):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.nbatch = self._get_nbatch()
        self.iter_counter = 0
        self.epoch_counter = 0
        self.criteria_raw = []
        self.criteria_summary = []
        self.stat_fun = stat_fun
        self.stop_rule = stop_rule
        # internal state for current epoch
        self._criteria_curr = self._empty_criteria_curr()  
        self._iter_in_epoch_counter = 0
    def update(self, step_size = 1):
        '''
        update the internal states and return -1 if still in the same epoch or 0 if update epoch
        '''
        self.iter_counter += step_size
        epoch_now = self._get_curr_epoch()
        if epoch_now == self.epoch_counter:
            self._iter_in_epoch_counter += 1
            return -1
        elif epoch_now == self.epoch_counter + 1:
            self._iter_in_epoch_counter = 0
            self.epoch_counter += 1
            return 0
    def record(self, update_return, criteria):
        '''
        it should only be run after running update
        '''
        self.criteria_raw.append([self.iter_counter, self.epoch_counter, criteria])
        self._criteria_curr[self._iter_in_epoch_counter] = criteria
        if update_return == -1:
            pass
        elif update_return == 0:
            self.criteria_summary.append(self.stat_fun(self._criteria_curr))
            self._criteria_curr = self._empty_criteria_curr()
            return 0
    def ifstop(self):
        return self.stop_rule(self.criteria_summary)
    def _empty_criteria_curr(self):
        return np.zeros((self.nbatch, ))
    def _get_nbatch(self):
        div = int(self.sample_size / self.batch_size) 
        if self.sample_size % self.batch_size == 0:
            return div
        else:
            return div + 1
    def _get_curr_epoch(self):
        return int(self.iter_counter / self.sample_size)


