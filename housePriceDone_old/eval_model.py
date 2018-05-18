from reusable import io


def eval_model_on_cv(df):
    df = clean_data(df)
    df = add_features(df)
    model = get_lgbmodel(n_jobs=10, n_estimators=9999)
    fitter = CVFitter(model)
    ixs = get_cv_ixs(df)
    result = fitter.fit(mm, targets, ixs)
    preds = result['combined_preds']
    val_ixs = get_val_ixs(ixs)
    evals = evaluate_loss(preds, df)
    io.append_csv(evals, results_path)


