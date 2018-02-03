from src.analysis import stationary_tests, plot_rolling_stats, plot_series, plot_auto_correlation
from src.preprocessing import difference, trend_smoothing, trend_regression, seasonal_regression
from src.utils import data_utils as du
from src.utils.experiments_utils import expr_sub_dir, expr_file_name
from matplotlib import pyplot as plt
import os

ANALYSIS_RESULTS_DIR = expr_sub_dir('python-analysis')


def main():
    data = du.get_data_dict()
    output = True

    for country, data_frame in data.items():
        for variable in data_frame.columns:
            series = data[country][variable]
            original_series = series
            original_series.name = original_series.name + ' original'

            for series in [original_series, difference(series), trend_regression(series)[0],
                           seasonal_regression(series)[0], trend_smoothing(series)[0]]:
                file_path = os.path.join(ANALYSIS_RESULTS_DIR, country + '_' + series.name.replace(' ', '_'))

                plot_series(series)
                plt.savefig(file_path + '_plot.png')

                plot_rolling_stats(series)
                plt.savefig(file_path + '_roll.png')

                res = stationary_tests(series)
                res.to_csv(file_path + '_test.csv', float_format='%.3f')

                plot_auto_correlation(series)
                plt.savefig(file_path + '_ACF.png')


if __name__ == '__main__':
    main()
