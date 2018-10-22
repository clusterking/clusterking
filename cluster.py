#!/usr/bin/env python3

"""Read the results from distance_matrix.py and cluster them.
"""

import argparse
import os.path
import sys

from modules.util.log import get_logger
from modules.util.cli import yn_prompt


log = get_logger("Cluster")


def read_input(input_file):
    pass


def write_out_clusters(df, path):
    pass


def cli():
    """Command line interface to run the integration jobs from the command
    line with additional options.

    Simply run this script with '--help' to see all options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="Input file/distance matrix.",
                        default="output/distance/global_results.out",
                        dest="input_path")
    parser.add_argument("-o", "--output",
                        help="Output file.",
                        default="output/cluster/global_results.out",
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
    write_out_clusters(df, args.output_path)

if __name__ == "__main__":
    # Run command line interface
    cli()