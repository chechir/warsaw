target_name = 'SalesPrice'


def generate_kaggle_submission(df_train, df_test, submission_file):
    df = append_dfs(df_train, df_test)
    df = clean_data(df)
    df = add_features(df)
    model = get_lgbmodel(n_jobs=10, n_estimators=30)
    ixs = get_lb_ixs(df)

    mm = get_mm_df(df)
    targets = get_targets(df)

    mm_train, train_targets = mm[ixs], targets[ixs]
    model.fit(mm_train, train_targets)

    mm_test = get_mm_df(df_test)
    preds = model.predict(mm_test)
    create_submission_file(df_test, preds)
