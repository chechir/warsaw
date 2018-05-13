from collections import OrderedDict
import copy
import numpy as np
from utils import io


class CVFitter():
    def __init__(self, model, save_dir=None):
        self.model = model
        self.save_dir = save_dir
        self._save_results = save_dir is not None
        if self._save_results:
            io.ensure_dir_exists(self.save_dir)

    def fit(self, mm, targets, ixs, **fitting_kwargs):
        assert isinstance(ixs, OrderedDict)
        results = OrderedDict()
        early_stopping = 'early_stopping_rounds' in fitting_kwargs
        for fold_name, fold_ix in ixs.iteritems():
            train_ix, val_ix = fold_ix['train'], fold_ix['val']
            mm_train, targets_train = mm[train_ix], targets[train_ix]
            mm_val, targets_val = mm[val_ix], targets[val_ix]
            fold_model = copy.deepcopy(self.model)
            if early_stopping:
                fitting_kwargs['early_stopping_data'] = {
                        'mm': mm_val,
                        'targets': targets_val
                        }
            fold_model.fit(mm_train, targets_train, **fitting_kwargs)
            preds = fold_model.predict(mm_val)
            results[fold_name] = (fold_model, preds)

        combined = np.concatenate([p for _, p in results.values()])
        results['combined_preds'] = combined
        return results