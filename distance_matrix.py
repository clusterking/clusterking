#!/usr/bin/env python3

"""Read the results from scan.py (q2 distributions) and calculate a distance
matrix.
"""

import argparse
import os.path
import pandas as pd
import sys

from modules.util.log import get_logger
from modules.util.cli import yn_prompt
from modules.inputs import Wilson

log = get_logger("Matrix")


def read_input(in_path):

    log.debug("Reading input.")

    with open(in_path, "r") as in_file:
        data = pd.io.json.loads(in_file.read(), orient="split")


    in_config = data["config"]
    in_df = pd.DataFrame(data["data"])

    in_df.sort_index(inplace=True)

    print(in_df)

    log.debug("Finished reading input.")

    sys.exit(0)

    return in_config, in_df


# todo: This should probably be implemented in a different way in the future,
# ideally separating the calculation of the matrix from the way we format the
# output
def calculate_matrix(config, in_df):



    # dirname = os.path.dirname(out_path)
    # if dirname and not os.path.exists(dirname):
    #     os.makedirs(dirname)

    # q2_values = config["bin_edges"]
    #
    # bpoints = int(len(df))
    # sep = "  "

    w = Wilson(0, 0, 0, 0, 0)

    rows = []
    for index_row_1, row_1 in in_df.iterrows():
        for index_row_2, row_2 in in_df.iterrows():

            print(row_1, row_2)
            # rows.append(row_1.update(row_))
            # pass
            # print(index_row_1, index_row_2)
            # print(in_row_1.values(), in_row_2.values())

    return
    cols = [ key+"_1" for key in w.dict().keys() ]
    cols.extend([ key+"_2" for key in w.dict().keys() ])
    cols.append("matrix_element")
    out_df = pd.DataFrame(columns=cols)

    # # todo: This is a very fragile way to read and write out data
    # with open(out_path, "w") as outfile:
    #     for j in range(1, bpoints):
    #         for i in range(j, bpoints):
    #             df1 = df[(i-1)*qpoints : i*qpoints].reset_index()
    #             df2 = df[(j-1)*qpoints : j*qpoints].reset_index()
    #             chi2 = sum((df1.dist - df2.dist )**2)
    #             outfile.write("{}{}{}{}{:.4f}\n".format(i, sep, j, sep, chi2))


def cli():
    """Command line interface to run the integration jobs from the command
    line with additional options.

    Simply run this script with '--help' to see all options.
    """
    desc = "Read the results from scan.py (q2 distributions) and calculate " \
           "a distance matrix."

    parser = argparse.ArgumentParser(description=desc)
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

    log.info("Starting to calculate matrix.")

    config, df = read_input(args.input_path)
    print(config)
    print(df)
    matrix = calculate_matrix(config, df)

    log.info("Finished.")

if __name__ == "__main__":
    # Run command line interface
    cli()
