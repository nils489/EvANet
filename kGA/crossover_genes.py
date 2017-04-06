#!/usr/bin/env python

import numpy as np
import pickle
import sys

#read gene-files
gene1_filename = sys.argv[1]
gene2_filename = sys.argv[2]
gene1_file = open(gene1_filename, 'rb')
gene2_file = open(gene2_filename, 'rb')

gene1 = pickle.load(gene1_file)
gene2 = pickle.load(gene2_file)

gene1_file.close()
gene2_file.close()

# randomly determine crossover point
crossover_point = np.random.randint(len(gene1))
print("crossing over at "+str(crossover_point))

for i in range(crossover_point, len(gene1)):
	gene1[i] = gene2[i]

for i in range(0, crossover_point):
	gene2[i] = gene1[i]


# save new genes
gene1_file = open(gene1_filename, 'wb')
gene2_file = open(gene2_filename, 'wb')
pickle.dump(gene1, gene1_file)
pickle.dump(gene2, gene2_file)
gene1_file.close()
gene2_file.close()

