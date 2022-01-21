import pandas as pd
import numpy as np
import os
import argparse
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    repo_dir = os.getcwd() + '/repo'
    tft_dir = os.path.join(repo_dir, 'tft')
    os.chdir(tft_dir)

    output_folder = os.path.join(os.getcwd(), args.output_folder)
    force_download = False
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    datasets_dir = {'MobiAct': ['mobiact_dataset/', 'mobiact_preprocessed/']
                    }

