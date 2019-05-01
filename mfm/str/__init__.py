import re


def clean(s):
    """ Remove special characters to clean up string and make it compatible
    with a Python variable names

    :param s:
    :return:
    """
    s = re.sub('[^0-9a-zA-Z_]', '', s)
    # Remove leading characters until we find a letter or underscore
    s = re.sub('^[^a-zA-Z_]+', '', s)
    return s