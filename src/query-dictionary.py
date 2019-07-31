#!/usr/bin/env python3

import time, sys
import numpy as np
from os import listdir
from collections import namedtuple
import pickle

DictElem = namedtuple("DictElem", "index file")

from hdlib.pyhdlib import hd_encode as hd
from util_lib import ProgressBar

# init HD encoder
ngramm = 3
encoding = "sumNgramm"
nitem = 26
D = 10000
SPARSITY = 5
device = 'cpu'


def checkInclusion(partial_index, index):
    for i, bit in enumerate(partial_index):
        if bit == 1:
            if bit != index[i]:
                return 0
    return 1


def encodeQuery(encoder, query, file):
    partial_index = encoder.encodeText(query)
    partial_index = [bit.item() for bit in partial_index]
    fp = open(file, 'w+')
    fp.write(''.join([str(bit) for bit in partial_index]))
    fp.close()
    return partial_index

def queryDictionary(dict, partial_index):
    matches = []

    bar = ProgressBar(len(dict), 'Query progress...')
    for i, entry in enumerate(dict):
        included = checkInclusion(partial_index, [bit.item() for bit in entry.index])
        if included:
            matches.append(entry.file)
        bar.update(i)

    return matches


def reportQuery(matches, report_f):
    rep = open(report_f, 'w+')
    for match in matches:
        f = open(match, "r")
        # rep.write("{},'{}'\n".format(match, f.read()))
        rep.write("{}\n".format(match))
        f.close()
    rep.close()

def main():
    # Command-line arguments are:
    # D, item memory input file, dict input file, output files path
    argcount = 6
    if len(sys.argv) != (argcount+1):
        print("Program requires {} arguments: <D> <SPARSITY> <item mem path> <dict input file> <output file path> <query>".format(argcount))
        return 0
    global D
    global SPARSITY
    D = int(sys.argv[1])
    SPARSITY = int(sys.argv[2])
    itemmem_file = sys.argv[3]
    dict_file = sys.argv[4]
    o_filepath = sys.argv[5]
    query = sys.argv[6]

    encoder = hd.hd_encode(D, encoding, device, nitem, ngramm, SPARSITY, 0, itemmem_file)

    # Load dictionary
    with open(dict_file, 'rb') as fp:
        dict = pickle.load(fp)
        fp.close()

    print("Searching query '{}'...".format(query))
    enc_query = encodeQuery(encoder, query, o_filepath + '/query.txt')
    matches = queryDictionary(dict, enc_query)

    reportQuery(matches, o_filepath+"/query_report.txt")


if __name__ == '__main__':
    main()
