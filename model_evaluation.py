'''Entrypoint'''

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fnmatch import fnmatch
from os import listdir
from os.path import join as pathjoin, dirname as pathdirname

from reporter import Reporter

def setup_parser(parser):
    '''Set arguments for parsing'''
    parser.add_argument(
        '-o', '--output-filename', type=str, default='exp0',
        help='output file name without extension, it is used for saving reporting results'
    )
    parser.add_argument(
        '-p', '--prediction-list', nargs='+',
        help='absolute/relative prediction filepaths list' \
             ', type only filename for taking file from the "predictions" dir' \
             ', start with "./" for a path relative for the current workdir' \
             ', by default the whole "predictions" dir will be considered'
    )

def construct_path_from(argpath, filetype, rel_path_to_this_script_dir):
    if isinstance(argpath, list):
        return [construct_path_from(path, filetype, rel_path_to_this_script_dir)
                for path in argpath]
    if '/' in argpath:
        return argpath
    if '*' in argpath:
        files = []
        for filename in listdir(pathjoin(rel_path_to_this_script_dir, filetype)):
            if fnmatch(filename, argpath):
                files.append(construct_path_from(filename, filetype, rel_path_to_this_script_dir))
        return files
    return pathjoin(rel_path_to_this_script_dir, filetype, argpath)

def main(args):
    '''Entrypoint'''
    rel_path_to_this_script_dir = pathdirname(args[0])
    rel_path_to_this_script_dir = rel_path_to_this_script_dir or '.'

    parser = ArgumentParser(
        description='Model evaluation framework',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    parsed_args = parser.parse_args(args[1:])

    output_filepath_template = construct_path_from(parsed_args.output_filename
                                                       , 'output'
                                                       , rel_path_to_this_script_dir)
    predictions_list = parsed_args.predictions_list
    if predictions_list is None:
        predictions_list = '*'
    predictions_list = construct_path_from(predictions_list
                                           , 'predictions'
                                           , rel_path_to_this_script_dir)

    reporter = Reporter(output_filepath_template)
    reporter.report(predictions_list)


if __name__ == '__main__':
    import sys
    main(sys.argv)
