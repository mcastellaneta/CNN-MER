# Sort the list in a specific way, so that 'x_100' goes after 'x_2'
# instead of lexycographic order
# SOURCE: https://nedbatchelder.com/blog/200712/human_sorting.html

import re


def tryint(s):
    """
    Return an int if possible, or `s` unchanged.
    """
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.

    >>> alphanum_key("z23a")
    ["z", 23, "a"]

    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def human_sort(list):
    """
    Sort a list in the way that humans expect.
    """
    list.sort(key=alphanum_key)