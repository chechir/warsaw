import lightgbm


class LGBRModel():
    internal_model = lightgbm.LGBMRegressor
    mm_columns = None

    def __init__(self, params):
        self.model_params = params
        self._model = self.internal_model(**self.model_params)

    def fit(self, mm, targets, *args, **kwargs):
        self.mm_columns = mm.columns
        self._model.fit(mm.values, targets, *args, **kwargs)

    def predict(self, mm, *args, **kwargs):
        predictions = self._predict(mm, *args, **kwargs)
        return predictions

