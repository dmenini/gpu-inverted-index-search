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

# Return path of all files to process
def load_files(path):
    files = listdir(path)
    # italian_files = [file for file in files if file[0:2]=='it']
    # print("Ita: {}/{}".format(len(italian_files), len(files)))
    return [path + "/" + file for file in files]

def count_zeroes(index):
    count = 0
    for bit in index:
        if bit == 0:
            count += 1
    return count


def exportDictionaryToCSV(file, dict):
    with open(file, mode='w') as fp:
        for entry in dict:
            fp.write(entry.file)
            fp.write(',')
            for bit in entry.index:
                fp.write(bit)
            fp.write('\n')
        fp.close()


def generate_dictionary(encoder, files):

    encoder = hd.hd_encode(D, encoding, device, nitem, ngramm, sparsity, resolution)
    dictionary = []

    for i, file in enumerate(files):
        f = open(file, "r")
        # compute index hypervector
        text = f.read()
        # print("Text: '{}'".format(text))
        index_hv = encoder.encodeText(text)
        print(count_zeroes(index_hv))
        dictionary.append(DictElem(index=index_hv, file=file))
        f.close()

    return dictionary, encoder



def main(D, i_filepath, o_filepath):
    files = load_files(i_filepath)
    dict, enc = generate_dictionary(encoder, files)
    enc.exportItemMemory(o_filepath+"/itemmemory.bin")
    enc.exportItemMemoryToCSV(o_filepath+"/itemmemory.csv")
    exportDictionaryToCSV(o_filepath+'/dictionary.csv', dict)

    # export dictionary also as binary
    with open(o_filepath+'/dictionary.bin', 'wb') as fp:
        pickle.dump(dict, fp)
        fp.close()

if __name__ == '__main__':
    # Command-line arguments are:
    # D, input files path, output files path
    argcount = 3
    if sys.argc != argcount:
        print("Program requires {} arguments: <D> <input file path> <output file path>".format(argcount))
        return 0
    global D
    D = int(sys.argv[0])
    i_filepath = sys.argv[1]
    o_filepath = sys.argv[2]
    main(D, i_filepath, o_filepath)
    return 1

