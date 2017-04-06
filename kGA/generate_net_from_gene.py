#!/usr/bin/env python

import numpy as np
import pickle
import sys
import re

out_filename = sys.argv[1]
chrom_size = 63
allel_size = 3
file_list = []
lines_list = []
out_list = []
file_list.append(open('templates/000_template.prototxt', 'r'))
file_list.append(open('templates/001_template.prototxt', 'r'))
file_list.append(open('templates/010_template.prototxt', 'r'))
file_list.append(open('templates/011_template.prototxt', 'r'))
file_list.append(open('templates/100_template.prototxt', 'r'))
file_list.append(open('templates/101_template.prototxt', 'r'))
file_list.append(open('templates/110_template.prototxt', 'r'))
file_list.append(open('templates/111_template.prototxt', 'r'))

# helper function to translate indices
def translate_allel_to_index(al):
	return al[0]*4+al[1]*2+al[2]

# read all template files
for f in file_list:
	lines_list.append(f.readlines())
	f.close()

gene_filename = sys.argv[2]
gene_file = open(gene_filename, 'rb')
out_file  = open(out_filename, 'w')

# append the initial net structure
for j in range(0,117):
	out_list.append(lines_list[3][j])
allel_chrom = pickle.load(gene_file)
gene_file.close()

#append blocks, according to the individuals chromosome
for i in range(0, len(allel_chrom)):
	tmp_str = "Block "+str(i+1)+" -"
	pattern = re.compile(tmp_str)
	tmp_str = "Block "+str(i+2)+" -"
	end_pattern = re.compile(tmp_str)
	for k in range(0, len(lines_list[translate_allel_to_index(allel_chrom[i])])):
		if(pattern.search(lines_list[translate_allel_to_index(allel_chrom[i])][k])):
			st_ind = k
	for k in range(0, len(lines_list[translate_allel_to_index(allel_chrom[i])])):
		if(end_pattern.search(lines_list[translate_allel_to_index(allel_chrom[i])][k])):
			en_ind = k
	for l in range(st_ind, en_ind):
		out_list.append(lines_list[translate_allel_to_index(allel_chrom[i])][l])
#append the final part of the net
for j in range(6793,6891):
	out_list.append(lines_list[3][j])
out_file.writelines(out_list)
out_file.close()



