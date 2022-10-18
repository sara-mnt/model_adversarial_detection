import ast
import os
from abc import abstractmethod


def multiple_keys(d,keys):
    ret=d
    for k in keys:
        ret=ret[k]
    return ret


def guess_convert(s):
    if s is None:
        return None
    try:
        value = ast.literal_eval(s)
    except Exception:
        return s
    else:
        return value


class Init(object):
    def __getattr__(self, k):
        return self.get(k, None)

    @abstractmethod
    def get(self, k, default=None):
        pass


class DictInit(Init):
    def __init__(self, data):
        self.data = data

    def get(self, k, default=None):
        try:
            return multiple_keys(self.data, k.split("."))
        except:
            return default


class EnvInit(Init):
    def __init__(self, conv=guess_convert):
        self.conv = conv

    def get(self, k, default=None):
        temp = os.environ.get(k, default)
        if self.conv:
            temp = self.conv(temp)
        return temp
