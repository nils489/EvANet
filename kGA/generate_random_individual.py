#!/usr/bin/env python

import numpy as np
import pickle
import sys

out_filename = sys.argv[1]
chrom_size = 63
allel_size = 3
allel_list = []

wfile = open(out_filename, 'wb')

# generate random bitstring
bstr = np.random.randint(2, size=(chrom_size,))

# partition bitsring to allels
i = 0
while i < chrom_size:
	allel_list.append(bstr[i:i+allel_size]) 
	i += allel_size
pickle.dump(allel_list, wfile)

wfile.close()
