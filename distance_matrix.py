#!/usr/bin/env python3

"""Read the results from scan.py (q2 distributions) and calculate a distance
matrix.
"""

import argparse
import os.path
import pandas as pd
import sys

from modules.util.log import get_logger
from modules.util.misc import yn_prompt

log = get_logger("Matrix")


def write_out_matrix(df, path):
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    q2_values = df.q2.unique()

    if not len(df) % len(q2_values) == 0:
        log.critical("Input file seems to be corrupted.")

    qpoints = len(q2_values)
    bpoints = int(len(df) / qpoints)

    sep = "  "

    # todo: This is a very fragile way to read and write out data
    with open(path, "w") as outfile:
        for j in range(1, bpoints):
            for i in range(j, bpoints):
                df1 = df[(i-1)*qpoints : i*qpoints].reset_index()
                df2 = df[(j-1)*qpoints : j*qpoints].reset_index()
                chi2 = sum((df1.dist - df2.dist )**2)
                outfile.write("{}{}{}{}{:.4f}\n".format(i, sep, j, sep, chi2))


def read_input(path):
    columns = ["epsL", "epsR", "epsSR", "epsSL", "epsT", "q2", "dist"]
    df = pd.read_csv(path,
                     sep='\s+',
                     header=None,
                     names=columns)

    return df


def cli():
    """Command line interface to run the integration jobs from the command
    line with additional options.

    Simply run this script with '--help' to see all options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="Input file/q2 histograms.",
                        default="output/scan/global_results.out",
                        dest="input_path")
    parser.add_argument("-o", "--output",
                        help="Output file.",
                        default="output/distance/global_results.out",
                        dest="output_path")
    args = parser.parse_args()

    log.info("Input file: '{}'.".format(args.input_path))

    if os.path.exists(args.output_path):
        agree = yn_prompt("Output path '{}' already exists and will be "
                          "overwritten. Proceed?".format(args.output_path))
        if not agree:
            log.critical("User abort.")
            sys.exit(1)

    log.info("Output file: '{}'".format(args.output_path))

    df = read_input(args.input_path)
    write_out_matrix(df, args.output_path)

if __name__ == "__main__":
    # Run command line interface
    cli()