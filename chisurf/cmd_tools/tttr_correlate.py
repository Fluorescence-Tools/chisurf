#!/usr/bin/env python

from __future__ import annotations

import argparse
import numpy as np


def main():
    print("Correlate TTTR data")
    print("===================")
    print("TODO TODO")
    parser = argparse.ArgumentParser(
        description='Process some integers.'
    )
    parser.add_argument(
        'min_bin',
        metavar='lower',
        type=float,
        default=0.0,
        help='The lower end of the bins'
    )
    parser.add_argument(
        'max_bin',
        metavar='upper',
        type=float,
        default=100,
        help='The upper end of the bins'
    )
    parser.add_argument(
        'n_bins',
        metavar='n_bins',
        type=int,
        default=101,
        help='The number of bins'
    )
    parser.add_argument(
        'normalize',
        metavar='norm',
        type=bool,
        help='Normalize the histogram'
    )
    parser.add_argument(
        'filename',
        metavar='file',
        type=str,
        help='The filename used to generate the histogram'
    )

    args = parser.parse_args()
    print("\tFilename: %s" % args.labeling_file)
    print("")
    print("\tLower :\t%s" % args.min_bin)
    print("\tUpper :\t%s" % args.max_bin)
    print("\tn bins:\t%s" % args.n_bins)
    print("\tnorm  :\t%s" % args.normalize)

    data = np.loadtxt(args.labeling_file, unpack=True)
    print("Data-shape: %s" % data.shape)

    bins = np.linspace(args.min_bin, args.max_bin, args.n_bins)
    hist, edges = np.histogram(data, bins=bins, density=args.normalize)

    np.savetxt(
        args.labeling_file + '_hist.txt',
        np.vstack(
            (edges[:-1], hist)
        ).T,
        delimiter='\t'
    )


if __name__ == "__main__":
    main()

