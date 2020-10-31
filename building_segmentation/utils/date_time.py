import time
from datetime import datetime, timedelta


def get_time(time_in_second):
    sec = timedelta(seconds=int(time_in_second))
    d = datetime(1, 1, 1) + sec
    total_time = "{}:Days {}:Hours {}:Minutes {}:Seconds".format(
        d.day - 1, d.hour, d.minute, d.second
    )
    return total_time


def get_date():
    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    return date


def get_time_stamp():
    timestamp = time.strftime("%Y%m%dT%H%M", time.localtime())
    return timestamp
