import time


def get_datetime_str(format = "%m-%d_%H:%M:%S",time_slot = None):
    if time_slot is None:
        ts = time.strftime(format, time.localtime())
    else:
        ts = time.strftime(format, time_slot)
    return str(ts)


def format_sec_tohms(second=0):
    hour = int(second // 3600)
    minute = int(second % 3600 // 60)
    second1 = int(second % 60)
    return str(hour)+":"+str(minute)+":"+str(second1)