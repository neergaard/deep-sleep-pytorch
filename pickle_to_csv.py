""" Conversion of pickle files to comma-separated-values (csv) file. 
You can supply either a directory containing .pkl files via the '-d/--data_dir' flag, or
a single .pkl file using the '--file' flag.

Alexander Neergaard Zahid, 2021.
"""

import argparse
import time
from pathlib import Path
from tqdm import tqdm

import pandas as pd

from src.utils.pickle_reader import read_pickle


def convert_to_csv(filepath):
    predictions, targets = read_pickle(filepath)
    df = pd.concat(
        [pd.DataFrame(targets, columns=["Hypnogram"]), pd.DataFrame(predictions, columns=["W", "N1", "N2", "N3", "R"])],
        axis=1,
    )
    df.to_csv(filepath.parent / (filepath.stem + ".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--data_dir", type=str, help="Path to directory containing .pkl files to convert.")
    parser.add_argument("-f", "--file", type=str, help="Path to specific file to convert.")
    args = parser.parse_args()

    assert (args.data_dir is not None and args.file is None) or (
        args.data_dir is None and args.file is not None
    ), f"Specify either a data directory or a file, received data_dir={args.data_dir} and file={args.file}"

    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        list_files = sorted(list(data_dir.glob("**/*.pkl")))
        # N = len(list_files)
        N = 10
        list_files = list_files[:N]
        if N == 0:
            print(f"No .pkl files found in directory!")
        else:
            print(f"Starting conversion of {N} .pkl files...")
            bar = tqdm(list_files)
            start = time.time()
            for filepath in bar:
                bar.set_description(filepath.stem)
                convert_to_csv(filepath)
            end = time.time()
            print(f"Finished, {N} files converted in {end-start} seconds.")
    elif args.file is not None:
        N = 1
        filepath = Path(args.file)
        print(f"Converting file {filepath.stem}...")
        start = time.time()
        convert_to_csv(filepath)
        end = time.time()
        print(f"Converted {filepath.stem} in {end-start} seconds.")
