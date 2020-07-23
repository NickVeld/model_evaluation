'''Entrypoint'''

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fnmatch import fnmatch
from os import listdir
from os.path import join as pathjoin, dirname as pathdirname

from metrics.utils import get_metric
from reporter import Reporter


def setup_parser(parser):
    '''Set arguments for parsing'''
    parser.add_argument(
        '-m', '--metrics-list', nargs='+', required=True,
        help='metric names list' \
             ', type only name of a class from the modules inside "metrics"' \
             ', or type sklearn.metric_name where metric name' \
             ' is name of metric function form the sklearn'
    )
    parser.add_argument(
        '-o', '--output-filename', type=str, default='exp0',
        help='output file name without extension, it is used for saving reporting results'
    )
    parser.add_argument(
        '-p', '--predictions-list', nargs='+',
        help='absolute/relative prediction filepaths list' \
             ', type only filename for taking file from the "predictions" dir' \
             ', start with "./" for a path relative for the current workdir' \
             ', by default the whole "predictions" dir will be considered'
    )
    parser.add_argument(
        '-f', '--predictions-filter', nargs='+',
        help='helps to select one model from family' \
             ', type in format family_starting_tag:selected_model'
    )
    parser.add_argument(
        '-c', '--whole-country', action='store_true',
        help='set this flag in order to sum data over regions ' \
             'and set country as minimal location entity'
    )
    parser.add_argument(
        '-n', '--horizons-list', nargs='+',
        help='selection of specific horizons from predictions'
    )
    parser.add_argument(
        '-d', '--date-selector', nargs='+',
        help='anchor date(s) for forecasting plots'
    )
    parser.add_argument(
        '-D', '--compare-diff-with-actual', action='store_true',
        help='set this flag in order to ' \
             'plot differences of predictions with actual values'
    )

def construct_path_from(argpath, filetype, rel_path_to_this_script_dir
                        , predictions_filter=None):
    if isinstance(argpath, list):
        files = []
        for path in argpath:
            files += construct_path_from(path, filetype, rel_path_to_this_script_dir
                                         , predictions_filter=predictions_filter)
        return files

    if '/' in argpath:
        return [argpath]
    if '*' in argpath:
        files = []
        for filename in listdir(pathjoin(rel_path_to_this_script_dir, filetype)):
            if (not(fnmatch(filename, argpath))
                    or(filename.startswith('.')) or(filename.endswith('Test.csv'))):
                continue
            flag = False
            for family_name, selected_models in predictions_filter:
                if filename.startswith(family_name):
                    flag = True
                    for selected_model in selected_models:
                        if filename.startswith(selected_model):
                            flag = False
                            break
                    if flag:
                        break
            if flag:
                continue
            files += construct_path_from(filename, filetype, rel_path_to_this_script_dir
                                         , predictions_filter=predictions_filter)
        return files
    return [pathjoin(rel_path_to_this_script_dir, filetype, argpath)]

def main(args, rel_path_to_this_script_dir):
    '''Entrypoint'''
    parser = ArgumentParser(
        description='Model evaluation framework',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    parsed_args = parser.parse_args(args)

    output_filepath_template = construct_path_from(parsed_args.output_filename
                                                   , 'output'
                                                   , rel_path_to_this_script_dir)[0]
    predictions_list = parsed_args.predictions_list
    if predictions_list is None:
        predictions_list = '*'
    predictions_filter = parsed_args.predictions_filter
    if predictions_filter is None:
        predictions_filter = []
    else:
        filt_parser = lambda x: (x[0], x[1].split(','))
        predictions_filter = [filt_parser(filt.split(':')) for filt in predictions_filter]

    predictions_list = construct_path_from(predictions_list
                                           , 'predictions'
                                           , rel_path_to_this_script_dir
                                           , predictions_filter=predictions_filter)

    metrics_list = [get_metric(metric_name) for metric_name in parsed_args.metrics_list]

    reporter = Reporter(output_filepath_template, parsed_args.whole_country)
    return reporter.report(predictions_list, metrics_list
                           , horizons_list=parsed_args.horizons_list
                           , date_selector=parsed_args.date_selector)


if __name__ == '__main__':
    print('Generated files:'
          + str(main(sys.argv[1:], rel_path_to_this_script_dir=pathdirname(sys.argv[0]) or '.')))
