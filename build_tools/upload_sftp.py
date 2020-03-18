#!/usr/bin/env python

import pysftp
import glob
import argparse
import os

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
    with pysftp.Connection(
            host=hostname,
            username=username,
            password=password
    ) as sftp:
        print("Connection established ... ")
        print("Uploading: %s" % local_path)
        sftp.remove(remote_file)
        sftp.put(local_path, remote_file)

