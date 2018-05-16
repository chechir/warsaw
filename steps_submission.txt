

#####  Now, lets make a submission to Kaggle
# lets start by building another functional test that exercises the whole pipeline:

####   we create 2 global dfs, to simulate the train and test sets

######### MAKE A FUNCTIONAL FAILING TEST

df = DDF({
    'Id': np.arange(3),
    'MiscFeature': [np.nan, 'Gar2', 'Othr'],
    'MoSold': [2, 5, 9],
    'SalePrice': [1000, 20000, 400000]
    })

train_df = df.colslice(df.columns)
test_df = df.drop_columns(['SalePrice'])


def test_make_submission_file(tmpdir):
    sub_path = str(tmpdir) + 'sub.csv'
    model_params = {'min_data': 1, 'min_data_in_bin': 1}
    ms.make_submission_file(train_df, test_df, sub_path, extra_model_params=model_params)
    result_df = DDF.from_csv(sub_path)
    assert 'SalePrice' in result_df.columns
    assert 'Id' in result_df.columns
    assert len(result_df.columns) == 2

########## Make the test pass, by writting the required functions:
def make_submission_file(train_df, test_df, sub_path, extra_model_params={}):
    df = append_dfs(train_df, test_df)
    df = ev.clean_data(df)
    df = ev.add_features(df)
    mm, targets = ev.get_mm(df), ev.get_targets(df)
    ixs = get_lb_ixs(df)
    params = {'n_estimators': 1100, 'learning_rate': 0.01, 'silent': 0}
    params.update(extra_model_params)
    model = ev.get_model(**params)
    fitter = CVFitter(model)
    results = fitter.fit(mm, targets, ixs)
    preds = results['preds'][0]
    generate_sub_file(preds, test_df['Id'], sub_path)


############# UNIT tests

def test_append_dfs():
    result_df = ms.append_dfs(train_df, test_df)
    assert len(result_df) == len(train_df) + len(test_df)

## make the test pass
def append_dfs(df_train, df_test):
    df_test['SalePrice'] = np.nan
    result_df = df_train.append(df_test, axis=0)
    return result_df


############ test 

def test_get_lb_ixs():
    df = DDF({
        'col1': np.arange(5),
        'SalePrice': np.array([100.]*3 + [np.nan]*2)
        })
    ixs = ms.get_lb_ixs(df)
    expected = {
            0: {'train': np.array([True]*3 + [False]*2),
                'val': np.array([False]*3 + [True]*2)}
            }
    assert np.all(expected[0]['train'] == ixs[0]['train'])
    assert np.all(expected[0]['val'] == ixs[0]['val'])

## make the test pass
def get_lb_ixs(df):
    train_ixs = np.isfinite(df[ev.target_name])
    result = OrderedDict({0: {'train': train_ixs, 'val': ~train_ixs}})
    return result


############ test
def test_generate_sub_file(tmpdir):
    preds = np.arange(10)
    id_sub = np.arange(10)
    sub_file_path = tmpdir + 'file.csv'
    ms.generate_sub_file(preds, id_sub, sub_file_path)
    result_df = DDF.from_csv(sub_file_path)
    assert result_df.shape[1] == 2
    assert result_df.shape[0] == 10


## make the test pass
def generate_sub_file(preds, sub_id, sub_file_path):
    df = DDF({
        'Id': sub_id,
        ev.target_name: np.exp(preds)
        })
    df.to_csv(sub_file_path)

