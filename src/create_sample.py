from utils import *
import config
import argparse


path_dict=config.PATH_DICT
clents_file=config.CLIENTS_FILE

if not os.path.exists(clents_file):
    os.mkdir(clents_file)

parser=argparse.ArgumentParser()
parser.add_argument("client_id", help='id of client whoose data will be extracted')

args=parser.parse_args()

create_sample(args.client_id, path_dict, clents_file)
