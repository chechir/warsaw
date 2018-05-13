import os


def append_csv(data, path):
    from ddf import DDF
    assert path.endswith('.csv')
    to_log = data if isinstance(data, DDF) else DDF(data)
    if os.path.isfile(path):
        current_log = DDF.from_csv(path)
        current_log = current_log.append(to_log, axis=0)
    else:
        current_log = to_log
    current_log.to_csv(path)
