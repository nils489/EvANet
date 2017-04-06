#!/usr/bin/env python

import numpy as np
import pickle
import sys
import re

filename = sys.argv[1]
allel_size = 3

# helper function to translate indices
def translate_allel_to_index(al):
	return al[0]*4+al[1]*2+al[2]

# function that flips one bit in the chromosome
def mutate_gene(gene_ind, flat_chrom):
	if(flat_chrom[gene_ind] == 1):
		flat_chrom[gene_ind] = 0
	elif(flat_chrom[gene_ind] == 0):
		flat_chrom[gene_ind] = 1
	print("gene mutated at "+str(gene_ind))

gene_file = open(filename, 'rb')
chromosome = pickle.load(gene_file)
gene_file.close()

# initialize flattened chromosome with zeros
flat_chromosome = np.zeros(len(chromosome)*allel_size)

# fill flattened chromosome
for i in range(0,len(chromosome)):
	for j in range(0,allel_size):
		flat_chromosome[(i*allel_size)+j] = chromosome[i][j]

# determine loci to mutate
mut_prob_arr = np.random.uniform(0.0,1.0,len(flat_chromosome))
for i in range(0, len(flat_chromosome)):
	if (mut_prob_arr[i] <= (1.0/len(flat_chromosome))):
		mutate_gene(i, flat_chromosome)

# write flattened chromosome back to original chromosome
for i in range(0,len(chromosome)):
	for j in range(0,allel_size):
		chromosome[i][j] = flat_chromosome[(i*allel_size)+j]


gene_file = open(filename, 'wb')
pickle.dump(chromosome, gene_file)
gene_file.close()

