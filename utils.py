import datetime
import os
from PIL import Image, ImageStat
from concurrent.futures import ThreadPoolExecutor


def format_time(elapsed):
    """
    :param elapsed: time in seconds
    :return: formatted time string in hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
