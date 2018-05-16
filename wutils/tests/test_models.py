from ddf import DDF
import numpy as np
from wutils.models import LGBRModel


def test_LGBRmodel():
    df = DDF({
        'MoSold': [2., 2., 9.]*100,
        'feat2': [1., 2., 1.]*100,
        })
    targets = np.array(np.random.random(100*3))
    model = LGBRModel(params={'n_estimators': 1})
    model.fit(df, targets)
    preds = model.predict(df)
    assert len(preds) == len(df)


def test_LGBRmodel_with_early_stopping():
    df = DDF({
        'MoSold': [2., 2., 9.]*100,
        'feat2': [1., 2., 1.]*100,
        })
    targets = np.array(np.random.random(100*3))
    model = LGBRModel(params={'n_estimators': 1})
    early_stopping_data = {'mm': df, 'targets': targets}
    model.fit(df, targets, early_stopping_rounds=3,
              early_stopping_data=early_stopping_data)
    preds = model.predict(df)
    assert len(preds) == len(df)
