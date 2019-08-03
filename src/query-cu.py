#!/usr/bin/env python3

import os
import cffi
import timeit



def main():
    SETUP_CODE = '''
import os
import cffi

header = """
int main_py(int argc, char **argv);
void queryDictionary();
void finalize(char* report_file);
"""

ffi = cffi.FFI()
ffi.cdef(header)
package_directory = os.path.dirname(os.path.abspath(__file__))
lib_file = os.path.join(package_directory, '/home/sem19f29/hd-bitfunnel-gpu/bin/query.so')
C = ffi.dlopen(lib_file)

argv_l = [ffi.new("char[]", "./out/files.txt".encode()),
          ffi.new("char[]", "./out/inv_dictionary.int".encode()), 
          ffi.new("char[]", "./out/query.txt".encode()), 
          ffi.new("char[]", "./out".encode())]
argv = ffi.new("char *[]", argv_l)
argc = 4
report_f = ffi.new("char[]", "./out/c_query_report.txt".encode())
C.main_py(argc, argv)
    '''
    CODE = '''
C.queryDictionary()
    '''
    time = timeit.timeit(setup = SETUP_CODE, stmt = CODE, number=10000)
    print("CUDA Query takes: {}".format(time/10000))
    # C.queryDictionary()
    # C.reportQuery(report_f)


if __name__ == '__main__':
    main()
