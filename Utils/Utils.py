import re

def last_epoch_from_filename(filename: str):
    e = re.search(r'ckpt_(\d+)_.pt', filename)
    assert e.group != None, "match epoch failed."

    return int(e.group(1))