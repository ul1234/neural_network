#!/usr/bin/python
# -*- coding: utf-8 -*-

import pprint, sys

class Debug(object):
    ENABLE = False
    OUTPUT_FILE = ''
    _count = 0

    @classmethod
    def output_file(cls, filename):
        cls.OUTPUT_FILE = filename
        if filename: open(filename, 'w').close()

    @classmethod
    def print_(cls, *str):
        if cls.ENABLE:
            for s in str:
                pp = pprint.PrettyPrinter(width = 200, depth = 10, stream = open(cls.OUTPUT_FILE, 'a') if cls.OUTPUT_FILE else None)
                pp.pprint(s)
                sys.stdout.flush()

    @classmethod
    def count(cls):
        cls._count += 1
        return cls._count
        
