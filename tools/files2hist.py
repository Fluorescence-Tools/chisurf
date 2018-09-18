import argparse
from glob import glob
import numpy as np
import pandas as pd
import pylab as p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make a histogram for a given set of files.')
    parser.add_argument('filename', type=str,
                        help='The filename used to create a histogram. Here * can be used to pass multiple files,')
    parser.add_argument("-r", "--range", type=float, nargs="+", help="The range of the histogram.")
    parser.add_argument("-b", "--bins", type=int, help="The number of bins in the histogram.", default=101)
    parser.add_argument("-v", "--verbosity", type=int, help="increase output verbosity")
    parser.add_argument("-s", "--suffix", type=str, help="Suffix added to the filename", default=".hist.csv")
    parser.add_argument("-o", "--output", type=str, help="Output file to save the mean, standard deviation, and"
                                                         "the kurtosis of the data", default="dist_properties.csv")
    parser.add_argument("-c", "--column", type=int,
                        help="The column number which is used to generate the histogram. By default the first "
                             "column (=0) is used.",
                        default=0)
    parser.add_argument("-sep", "--separator", type=str,
                        help="Delimiter to use. If sep is None, will try to automatically determine this. "
                             "Separators longer than 1 character and different from 's+' will be interpreted "
                             "as regular expressions, will force use of the python parsing engine and will "
                             "ignore quotes in the data.",
                        default='\t')
    parser.add_argument("-head", "--header", type=int,
                        help="Row number(s) to use as the column names, and the start of the data. Default "
                             "behavior is as if set to 0 if no names passed, otherwise None. Explicitly pass "
                             "header=0 to be able to replace existing names. The header can be a list of "
                             "integers that specify row locations for a multi-index on the columns e.g. [0,1,3]. "
                             "Intervening rows that are not specified will be skipped (e.g. 2 in this example is "
                             "skipped). Note that this parameter ignores commented lines and empty lines if "
                             "skip_blank_lines=True, so header=0 denotes the first line of data rather than the "
                             "first line of the file.",
                        default=None)
    parser.add_argument("-skip", "--skiprows", type=int,
                        help="Line numbers to skip (0-indexed) or number of lines to skip (int) "
                             "at the start of the file",
                        default=0)
    parser.add_argument("-plot", "--plot", type=int,
                        help="Plots the histograms of the passed argument is bigger than 0.",
                        default=0)

    args = parser.parse_args()
    filenames = glob(args.filename)
    print filenames
    header = args.header
    with open(args.output, 'w') as fp:
        fp.write("Filename\tMean\tSD\tKurtosis\tSkewness\n")
        for filename in filenames:
            d = pd.read_csv(filename,
                            sep=args.separator,
                            header=args.header,
                            skiprows=args.skiprows
                            )
            ds = d[args.column]
            m = ds.mean()
            sd = ds.std()
            kurt = ds.kurtosis()
            skew = ds.skew()
            count, division = np.histogram(ds, range=args.range, bins=args.bins)
            if args.plot > 0:
                p.plot(division[1:], count)
            outfile = filename + args.suffix
            np.savetxt(outfile, np.vstack([division[1:], count]).T, delimiter=", ")#args.separator)
            fp.write("%s\t%s\t%s\t%s\t%s\n" % (filename, m, sd, kurt, skew))
        if args.plot > 0:
            p.show()
