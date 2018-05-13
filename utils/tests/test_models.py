from ddf import DDF
import numpy as np
from utils.models import LGBRModel


def test_LGBRmodel():
    df = DDF({
        'MoSold': [2., 5., 9.]*10,
        'SalePrice': [1000., 20000., 400000.]*10
        })
    model = LGBRModel(params={'n_estimators': 1})
    model.fit(df.colslice(['MoSold']), df['SalePrice'])
    preds = model.predict(df.colslice(['MoSold']))
    assert len(preds) == len(df)

