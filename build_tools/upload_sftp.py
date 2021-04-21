#!/usr/bin/env python

import pysftp
import glob
import argparse
import os

cnopts = pysftp.CnOpts()
cnopts.hostkeys = None

hostname = "ssh.fret.at"
username = "fret.at"
password = os.getenv('SFTP_PASSWD')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Update Apple Info.plist file with python module information.'
    )

    parser.add_argument(
        '-f',
        '--file_pattern',
        metavar='template',
        type=str,
        required=True,
        help='The file pattern. The first match is uploaded to the remote file, '
             'e.g, "../dist/*.exe"'
    )

    parser.add_argument(
        '-r',
        '--remote_file',
        metavar='template',
        type=str,
        required=True,
        help='The remote file, e.g., "./downloads/setup_daily.exe"'
    )
    args = parser.parse_args()

    file_pattern = args.file_pattern
    remote_file = args.remote_file
    local_path = glob.glob(file_pattern)[0]
    print("Uploading file: ", local_path)
    with pysftp.Connection(
            host=hostname,
            username=username,
            password=password,
            cnopts=cnopts
    ) as sftp:
        print("Connection established ... ")
        print("Uploading: %s" % local_path)
        try:
            sftp.remove(remote_file)
            print("File already on server. Deleting.")
        except FileNotFoundError:
            print("File not on server. Not necessary to delete.")
        sftp.put(local_path, remote_file)

