import logging
import pickle
import os

logger = logging.getLogger(__name__)


def pickle_dump(fn, var):
    logger.info(f"pickling {fn}, {len(var)}")
    # TODO: does it make sense to pickle by row, to save on memory but lose on IO?
    # https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(var)
    file_path = os.path.join('pickled_objects', fn)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def pickle_read(fn):
    logger.info(f"unpickling {fn}")
    # https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    file_path = os.path.join('pickled_objects', fn)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)
