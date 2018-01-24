import argparse
import os

from pylatex import Section, Figure
from pylatex.utils import NoEscape

COUNTRIES = ['EA', 'US']
VARIABLES = ['CPI', 'GDP', 'UR', 'IR', 'LR10', 'LR10-IR', 'EXRATE']


class Chapter(Section):
    def __init__(self, *args, **kwargs):
        super(Chapter, self).__init__(args, kwargs)
        self._latex_name = 'chapter'


def get_elements(dir):
    elements = {c: {v: [] for v in VARIABLES} for c in COUNTRIES}

    for i in os.listdir(dir):
        fname, ext = os.path.splitext(i)
        fpath = os.path.join(os.path.abspath(dir), fname)

        names = fname.split('_')
        if len(names) < 3:
            continue

        country = names[0]
        var = names[1]
        caption = ' '.join(names[2:])

        elements[country][var].append({'caption': caption, 'ext': ext, 'filename': fname, 'filepath': fpath})
    return elements


def main(input, output):
    elements = get_elements(input)

    main = Chapter('ARIMA Experiments', numbering=True)

    for country in COUNTRIES:
        for var in VARIABLES:
            with main.create(Section(' '.join([country, var]))) as subsec:
                for dic in elements[country][var]:
                    caption = dic['caption']
                    ext = dic['ext']
                    filepath = dic['filepath']
                    label = 'fig:' + dic['filename']

                    if ext != '.csv':
                        with subsec.create(Figure(position='h!')) as fig:
                            fig.add_image(filepath)
                            fig.add_caption(caption + " " + country + ' ' + var)
                            fig.append(NoEscape('\label{' + label + '}'))
                    elif ext == '.csv':
                        with subsec.create(Figure(position='h!')) as fig:
                            fig.append(NoEscape('\csvautotabular[respect all]{' + filepath + '.csv}'))
                            fig.add_caption(caption + " " + country + ' ' + var)
                            fig.append(NoEscape('\label{' + label + '}'))

            main.append(NoEscape("\clearpage"))

    if output is None:
        print main.dumps()
    else:
        main.generate_tex(output)


if __name__ == '__main__':
    descr = 'Script that parses directory of figures and tables and outputs' \
            ' a tex file with the code for importing them in a latex document.'

    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('input', action='store', help='Directory of figures and tables to be parsed.')
    parser.add_argument('-o', '--output', action='store', default=None, help='Output directory for tex file.')

    args = vars(parser.parse_args())

    main(args['input'], args['output'])
