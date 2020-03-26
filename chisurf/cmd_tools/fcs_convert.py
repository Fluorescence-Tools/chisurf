#!/usr/bin/env python

from __future__ import annotations

import sys
import argparse
import chisurf.fio
import chisurf.fio.fluorescence.fcs


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Convert FCS files.'
    )

    parser.add_argument(
        '-if',
        '--input_filename',
        type=str,
        required=True,
        help='The input FCS filename.'
    )

    parser.add_argument(
        '-it',
        '--input_type',
        type=str,
        required=True,
        help='The file type of the input FCS file either: '
             'kristine, alv, mat, confocor3, pycorrfit, csv, pq.dat'
    )

    parser.add_argument(
        '-of',
        '--output_filename',
        type=str,
        required=True,
        help='The output FCS filename.'
    )

    parser.add_argument(
        '-ot',
        '--output_type',
        type=str,
        required=True,
        help='The file type of the input FCS file either: '
             'kristine, alv, china-mat, confocor3, pycorrfit, pq.dat'
    )

    parser.add_argument(
        '-s',
        '--skiprows',
        type=int,
        default=0,
        required=False,
        help='Specifies the number of rows that are skipped in a csv file.'
    )

    parser.add_argument(
        '-e',
        '--use_header',
        type=bool,
        default=0,
        required=False,
        help='If this is set to True, the header is used to name columns, and'
             'skipped for reading the data columns.'
    )
    return parser.parse_args(args)


def main(
        args: argparse.Namespace = None
):
    if args is None:
        args = parse_args(sys.argv[1:])

    print("Convert FCS file")
    print("================")
    print("Input filename: %s" % args.input_filename)
    print("Input file type: %s" % args.input_type)
    print("Output filename: %s" % args.output_filename)
    print("Output file type: %s" % args.output_type)
    print("Writing files...")

    data = chisurf.fio.fluorescence.fcs.read_fcs(
        filename=args.input_filename,
        skiprows=args.skiprows,
        use_header=args.use_header,
        reader_name=args.input_type
    )
    chisurf.fio.fluorescence.fcs.write_fcs(
        data=data,
        filename=args.output_filename,
        file_type=args.output_type
    )
    print("")


if __name__ == "__main__":
    main()
