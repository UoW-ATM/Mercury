"""
This code is commented because of the compulsory networkx import that bloats requirement files.
"""

# from copy import copy
#
# import numpy as np
# import networkx as nx
#
#
# """
# That's python 3!!!
# """
#
# class DiMultiplex(nx.DiGraph):
# 	"""
# 	Class for dealing with multiplex network, based on the
# 	networkx directed graph.
# 	"""
#
# 	def __init__(self, layers=[]):
# 		super().__init__()
# 		self.layers = layers
#
# 	def add_node(self, n, layers=None, **kwargs):
# 		if layers == None:
# 			layers = self.layers
# 		else:
# 			for l in layers:
# 				self.layers.append(l)
# 		super().add_node(n, layers=layers, **kwargs)
#
# 	def has_node(self, n, layer=None):
# 		if layer==None:
# 			return super().has_node(n)
# 		else:
# 			return super().has_node(n) and layer in self.node[n]['layers']
#
# 	def nodes(self, layer=None):
# 		if not layer==None:
# 			return [n for n in super().nodes() if self.has_node(n, layer)]
# 		else:
# 			return super().nodes()
#
# 	def degree(self, n, layer=None):
# 		if layer != None:
# 			return super().degree(n, nbunch = self.nodes(layer))
# 		else:
# 			return super().degree(n)
#
# 	def in_degree(self, n, layer=None):
# 		if layer != None:
# 			return super().in_degree(n, nbunch = self.nodes(layer))
# 		else:
# 			return super().in_degree(n)
#
# 	def out_degree(self, n, layer=None):
# 		if layer != None:
# 			return super().out_degree(n, nbunch = self.nodes(layer))
# 		else:
# 			return super().out_degree(n)
#
# 	def add_edge(self, e1, e2, layers=None, attributes={}):
# 		"""
# 		CAREFUL: ALWAYS GIVE THE ATTRIBUTES KEYWORD!!!!
# 		"""
# 		if layers == None:
# 			layers = self.layers
# 		else:
# 			for l in layers:
# 				self.layers.append(l)
# 		if self.has_edge(e1, e2):
# 			for l in set(layers) - set(self[e1][e2].keys()):
# 				self[e1][e2][l] = attributes
# 				for e in [e1, e2]:
# 					if not l in self.node[e]['layers']:
# 						self.node[e]['layers'].append(l)
# 		else:
# 			self.add_node(e1)
# 			self.add_node(e2)
# 			super().add_edge(e1, e2)
# 			for l in layers:
# 				self[e1][e2][l] = copy(attributes)
#
# 	def get_attribute_value(self, e1, e2, name_attribute, layers=None, how='sum'):
# 		if layers == None:
# 			layers = self.layers
#
# 		coin = [self[e1][e2][lay][name_attribute] for lay in layers]
#
# 		if how=='sum':
# 			return sum(coin)
# 		elif how=='mean':
# 			return np.mean(coin)
#
# 	def edges(self, layer=None):
# 		if layer == None:
# 			return super().edges()
# 		elif layer == 'all':
# 			return [(e1, e2, l) for e1, e2 in super().edges() for l in self[e1][e2].keys()]
# 		else:
# 			return [e for e in super().edges() if self.has_edge(e[0], e[1], layer)]
#
# 	def has_edge(self, e1, e2, layer=None):
# 		if layer == None:
# 			return super().has_edge(e1, e2)
# 		else:
# 			return super().has_edge(e1, e2) and layer in self[e1][e2].keys()
#
# 	def neighbors(self, n, layer=None):
# 		if layer == None:
# 			return super().neighbors(n)
# 		else:
# 			return [nn for nn in super().neighbors(n) if self.has_edge(n, nn, layer = layer)]