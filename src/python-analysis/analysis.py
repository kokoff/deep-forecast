from code.analysis import stationary_tests, plot_rolling_stats, plot_series
from code.utils import data_utils as du
from code.utils.experiments_utils import expr_sub_dir, expr_file_name
from matplotlib import pyplot as plt
import os

ANALYSIS_RESULTS_DIR = expr_sub_dir('python-analysis')


def main():
    data = du.get_data()
    output = True

    # for country, data_frame in data.items():
    #     for variable in data_frame.columns:
    country = 'US'
    variable = 'LR10-IR'
    series = data[country][variable]

    file_path = os.path.join(ANALYSIS_RESULTS_DIR, country + '_' + variable)

    plot_series(series)
    plt.savefig(file_path + '_plot.png')

    plot_rolling_stats(series)
    plt.savefig(file_path + '_roll.png')

    res = stationary_tests(series)
    print '%.3f' % res['t-stat']['ADF']
    res['t-stat'].to_csv(file_path + '_test.csv', float_format='%.6f')

    # plt.show()


if __name__ == '__main__':
    main()
