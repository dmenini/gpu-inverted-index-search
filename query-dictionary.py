#!/usr/bin/env python3

import time, sys
import numpy as np
from os import listdir
from collections import namedtuple
import pickle

DictElem = namedtuple("DictElem", "index file")

from hdlib.pyhdlib import hd_encode as hd

# init HD encoder
ngramm = 3
encoding = "sumNgramm"
nitem = 26
D = 10000
sparsity = 99988
resolution = 100000
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

    for entry in dict:
        included = checkInclusion(partial_index, [bit.item() for bit in entry.index])
        if included:
            matches.append(entry.file)

    return matches


def printMatchedFiles(matches):
    for match in matches:
        f = open(match, "r")
        print("File: {}\tContent: '{}'\n".format(match, f.read()))
        f.close()

def main():
    # Command-line arguments are:
    # D, item memory input file, dict input file, output files path
    argcount = 5
    if len(sys.argv) != (argcount+1):
        print("Program requires {} arguments: <D> <item mem path> <dict input file> <output file path> <query>".format(argcount))
        return 0
    global D
    global N_FILES
    D = int(sys.argv[1])
    itemmem_file = sys.argv[2]
    dict_file = sys.argv[3]
    o_filepath = sys.argv[4]
    query = sys.argv[5]

    encoder = hd.hd_encode(D, encoding, device, nitem, ngramm, sparsity, resolution, 0, itemmem_file)

    # Load dictionary
    with open(dict_file, 'rb') as fp:
        dict = pickle.load(fp)
        fp.close()

    print("Searching query '{}'...".format(query))
    enc_query = encodeQuery(encoder, query, o_filepath + '/query.txt')
    matches = queryDictionary(dict, enc_query)
    printMatchedFiles(matches)


if __name__ == '__main__':
    main()
