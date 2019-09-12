from __future__ import annotations

import os
import pickle
import random


def get_fortune(
        fortunepath: str = None,
        min_length: int = 0,
        max_length: int = 100,
        attempts: int = 1000
):

    if fortunepath is None:
        fortunepath = os.path.abspath(__file__)

    fortune_files = [
        os.path.splitext(pdat)[0] for pdat in os.listdir(fortunepath) if pdat.endswith(".pdat")
    ]
    attempt = 0
    while True:
        fortune_file = os.path.join(fortunepath, random.choice(fortune_files))
        data = pickle.load(open(fortune_file+".pdat", "rb"))
        (start, length) = random.choice(data)
        print(random.choice(data))
        if length < min_length or (max_length is not None and length > max_length):
            attempt += 1
            if attempt > attempts:
                return ""
            continue
        with open(fortune_file, 'rU') as ffh:
            ffh.seek(start)
            fortunecookie = ffh.read(length)
        return fortunecookie

