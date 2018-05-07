import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


def plot_rolling_statistics(timeseries):
    stats = _get_rolling_stats(timeseries)
    stats['logerror'] = timeseries
    _plot_rolling_stats(stats)


def print_dickey_fuller_test(timeseries):
    raw_result = sm.tsa.adfuller(timeseries['abs_logerror'], autolag='AIC')
    result = _parse_raw_dickey_fuller_results(raw_result)
    print('Results of Dickey-Fuller Test:')
    print result()


def _get_rolling_stats(timeseries, past_days=31):
    stats = {}
    stats['rolling_mean'] = pd.rolling_mean(timeseries, window=past_days)
    stats['rolling_std'] = pd.rolling_std(timeseries, window=past_days)
    return stats


def _parse_raw_dickey_fuller_results(raw_result):
    indicators = [
            'Test Statistic', 'p-value',
            '#Lags Used', 'Number of Observations Used']
    result = pd.Series(raw_result[0:4], index=indicators)
    for key, value in raw_result[4].items():
        result['Critical Value (%s)' % key] = value
    return result


def _plot_rolling_stats(stats):
    plt.plot(stats['logerror'], color='blue', label='logerror')
    plt.plot(stats['rolling_mean'], color='red', label='Rolling Mean')
    plt.plot(stats['rolling_std'], color='black', label='Rolling Std')
    plt.legend(loc='best', fontsize=15)
    plt.figure(figsize=(15, 8))
    plt.title('Rolling Mean & Standard Deviation', fontsize=15)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Absolute Log Error', fontsize=15)
    plt.show(block=False)


