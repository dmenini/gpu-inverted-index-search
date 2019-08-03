#!/usr/bin/env python3

import time, sys
sys.path.append('/home/sem19f29/hd-bitfunnel-gpu')
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
N_FILES = 21000
SPARSITY = 9980
device = 'cpu'

# Return path of all files to process
def load_files(path):
    files = [file for file in listdir(path) if file.endswith('.txt')]
    return [path + "/" + file for file in files[0:N_FILES]]


# Count number of zeroes in each file encoding (high = sparse, thus is good).
def count_zeroes(index):

    count = 0
    for bit in index:
        if bit == 0:
            count += 1

    return count


# Export text names in an output file
def exportFiles(file, files):

    with open(file, mode='w') as fp:
        for entry in files:
            # for i in range(8):
            fp.write(entry + "\n") # str(i) + '_'
        fp.close()


# Export dictionary entries in an output file (CSV format)
def exportDictionaryToCSV(file, dict):

    with open(file, mode='w') as fp:
        for entry in dict:
            fp.write(entry.file)
            fp.write(',')
            for bit in entry.index:
                fp.write(str(bit.item()))
            fp.write('\n')
        fp.close()


# Export inverted dictionary entries in an output file (CSV format, useful for debug)
def exportInvertedDictionaryToCSV(file, dict):

    indeces = []

    for entry in dict:
        indeces.append(entry.index.tolist())

    mat = np.array(indeces)
    mat = np.transpose(mat)

    with open(file, mode='w') as fp:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                fp.write(str(mat[i, j].item()))
                if j != (mat.shape[1]-1):
                    fp.write(',')
                else:
                    fp.write('\n')
        fp.close()


# Convert from binary to decimal
def bin2dec(word):

    res = 0

    for i, bit in enumerate(word):
        res += 2**i * bit

    return res


# Export inverted dictionary entries in an output file (int format)
def exportInvertedDictionaryToInt(file, dict):

    indeces = []

    for entry in dict:
        indeces.append(entry.index.tolist())

    mat = np.array(indeces)
    mat = np.transpose(mat)
    word = []
    word_range = (mat.shape[1]//32)*32
    remainder_range = mat.shape[1]%32

    with open(file, mode='w') as fp:
        bar = ProgressBar(mat.shape[0], status='Inverting the dictionary...')
        for i in range(mat.shape[0]):  # mat.shape[0] = D
            for j in range(word_range):
                word.append(mat[i, j].item())
                if len(word) == 32:
                    word.reverse()
                    dec_word = bin2dec(word)
                    fp.write(str(dec_word) + ' ')
                    word.clear()
            for k in range(remainder_range):
                word.append(mat[i, word_range+k].item())
            word.reverse()
            dec_word = bin2dec(word)
            fp.write(str(dec_word) + '\n')
            word.clear()
            bar.update(i)
        fp.close()


# Export inverted dictionary entries in an output file (bin format)
def exportInvertedDictionary(file, dict):

    indeces = []

    for entry in dict:
        indeces.append(entry.index.tolist())

    mat = np.array(indeces)
    mat = np.transpose(mat)

    with open(file, mode='w') as fp:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                fp.write(str(mat[i, j].item()))
        fp.close()


# Generate dictionary (array of encoded entries)
def generate_dictionary(encoder, files, report):

    dictionary = []
    rep = open(report, 'w+')

    bar = ProgressBar(N_FILES, status='Generating the dictionary...')
    for i, file in enumerate(files):
        f = open(file, "r")
        rep.write(file+'\t')
        # compute index hypervector
        text = f.read()
        # print("Text: '{}'".format(text))
        index_hv = encoder.encodeText(text, rep)
        rep.write("{}\n".format(count_zeroes(index_hv)))
        # for k in range(8):
        filename = file # str(k) + '_' +
        dictionary.append(DictElem(index=index_hv, file=filename))
        f.close()
        bar.update(i)

    rep.close()

    return dictionary



def main():

    # Command-line arguments are:
    argcount = 7
    if len(sys.argv) != (argcount+1):
        print("Program requires {} arguments: <D> <SPARSITY> <N_FILES> <input file path> <output file path> <gen_item_mem> <generate_dictionary>".format(argcount))
        return 0
    global D
    global N_FILES
    global SPARSITY
    D = int(sys.argv[1])
    SPARSITY = int(sys.argv[2])
    N_FILES = int(sys.argv[3])
    i_filepath = sys.argv[4]
    o_filepath = sys.argv[5]
    gen_item_mem = int(sys.argv[6])
    gen_dict = int(sys.argv[7])

    dict = []
    files = load_files(i_filepath)

    enc = hd.hd_encode(D, encoding, device, nitem, ngramm, SPARSITY, gen_item_mem, o_filepath)
    
    dict_file = o_filepath+'/dictionary.bin'
    report_file = o_filepath+'/dict_report.txt'
    if gen_dict != 0:
        print("Generating dictionary...")
        dict = generate_dictionary(enc, files, report_file)
        exportFiles(o_filepath + '/files.txt', files)
        # exportDictionaryToCSV(o_filepath + '/dictionary.csv', dict)
        # Export dictionary as binary
        with open(dict_file, 'wb+') as fp:
            pickle.dump(dict, fp)
            fp.close()
    else:
        with open(dict_file, 'rb') as fp:
            print("Loading dictionary from '{}'...".format(dict_file))
            dict = pickle.load(fp)
            fp.close()

    exportInvertedDictionaryToInt(o_filepath + '/inv_dictionary.int', dict)
    # exportInvertedDictionaryToCSV(o_filepath + '/inv_dictionary.csv', dict)

    return 1

if __name__ == '__main__':
    main()