# EvANet
Residual Neural Networks, optimized by using a genetic algorithm

This repository contains some utility scripts and the implemented heterogeneous network structures from my master's thesis.

### abstract
In recent years, neural networks have been widely used as the method of choice for classification tasks in computer vision and image processing. The networks that currently provide the best results are very deep residual Convolutional Neural Networks. New network structures, based on the previously used topologies are developed each year, using different approaches.
In this thesis the application of a canonical genetic algorithm to optimize the topology of neural networks is investigated. Current state of the art networks are used as a baseline to start the optimization from. The generated architectures achieve an improved classification accuracy, compared to reference networks.
The developed network structures differ from commonly used architectures in their higher level architecture. In contrast to the reference networks they are not homogeneously constructed from repeating elements but heterogeneously from a sequence of different blocks. The improved classification accuracy suggests that the use of heterogeneous structures may be advantageous.
Furthermore it is shown that the generated neural networks converge in fewer iterations, producing better results.

### Citation
author, title, school, year

If you decide to use the developed structures in your research, please cite:

	@mastersthesis{korn_nl2017,
		author = {Nils Kornfeld},
		title = {Optimierung eines neuronalen Netzes zur Objekterkennung unter Verwendung evolutionärer Algorithmen},
		school = {Freie Universit\"at Berlin},
		year = {2017}
	}
  
  These structures are based on work from: https://github.com/KaimingHe/deep-residual-networks
