import os
import argparse
from end2end_framework import run

def run4all(input_dir, output_dir, weight, graphs=True, write='stats', write_file='ouput_stats.csv'):   
    """
    This function recursively calls the main function of the end2end_framework.py for a list of dicom files, e.g. an entire experiment.
    Parameters
    ----------
        input_dir: string
            The directory where the files are stored
        output_dir: string
            The directory in which to save the results
        weight: int
            The weight of the mice we are evaluating; we assume that all mice come from the same experiment so have the same weight
        graphs: bool, default=True
            If true graphs will be created and saved in the results' directory of each mouse, a subdirectory of output_dir
        write: string, default = 'stats'
            If 'stats' then values such as max, min, median etc of the LVIDs etc for each file are written to a csv file
            If 'all' then LVIDs etc. are written for all good classified regions
        write_file: string, default = 'ouput_stats.csv'
            The csv file to write results to according to what has been given to write 
    """ 
    list_files = [file for file in os.listdir(input_dir) if file.endswith('.dcm')]
    for file in list_files:
        print('Starting for file: ', file)
        file_name = file.split('.')[0]
        run(os.path.join(input_dir, file), os.path.join(output_dir, file_name), weight, graphs=graphs, write=write, write_file=os.path.join(output_dir, write_file))
        print('Done for file: ', file)
    print('Done for directory: ', input_dir.split('/')[-1])

# see the readme for further explanation of the arguments
def get_args():
    '''
    Required arguments
    ------------------
        -i: The path to the directory of files you wish to extract features from
        -m: The body mass in grams of the current mice you wish to extract features from
    Optional arguments
    ------------------
        -o: The name of the root directory to save graphs, images and csv to. Default is the current working directory. Note that for each
            file a subdirectory will be created if specified.
        -g: if True output graphs and images will be created and saved, if False they will not. Default is True
        -w: If 'all' all features are extracted and saved to a csv, if 'stats' only statistics from echocardiogram are extracted and saved 
            (one row per image). Default value is 'all'
    '''
    parser = argparse.ArgumentParser(description='Predict heart volume for test image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', metavar='INPUT', required=True,
                        help='Specify path directory of input files')
    parser.add_argument('--mass', '-m', type=int, required=True,
                        help='Specify the crop size')
    parser.add_argument('--output', '-o', default='.', 
                        help='Specify output path to save results')
    parser.add_argument('--graphs', '-g', default=False,
                        help='specify if you want to save graphs as output or not')       
    parser.add_argument('--write', '-w', default='all',
                        help='Specify if you want to write out all outputs or only stats')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.write=='stats':
        write_file = 'output_stats.csv'
    elif args.write=='all':
        write_file = 'output_all.csv'
    else:
        write_file = None
    run4all(args.input, args.output, args.mass, graphs=args.graphs, write=args.write, write_file=write_file)