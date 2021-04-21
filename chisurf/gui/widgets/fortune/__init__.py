from __future__ import annotations

import os
import pickle
import random


def get_fortune(
        fortunepath: str = None,
        min_length: int = 0,
        max_length: int = 100,
        maximum_number_of_attempts: int = 1000
):

    if fortunepath is None:
        fortunepath = os.path.dirname(__file__)

    fortune_files = [
        os.path.splitext(pdat)[0] for pdat in os.listdir(
            fortunepath
        ) if pdat.endswith(".pdat")
    ]
    attempt = 0
    while True:
        fortune_file = random.choice(fortune_files)
        print(fortune_file)
        with open(
                file=os.path.join(
                    fortunepath,
                    fortune_file + ".pdat"
                ),
                mode="rb"
        ) as fp:
            data = pickle.load(fp)
            (start, length) = random.choice(data)
            if length < min_length or (
                    max_length is not None and length > max_length
            ):
                attempt += 1
                if attempt > maximum_number_of_attempts:
                    return ""
                continue
        with open(
                file=os.path.join(
                    fortunepath,
                    fortune_file
                ),
                mode='rU'
        ) as ffh:
            ffh.seek(start)
            fortunecookie = ffh.read(length)
        return fortunecookie

