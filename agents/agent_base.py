from copy import copy

from Mercury.core.delivery_system import LetterBox


class Agent:
	"""
	Base Agent from which all Agents in Mercury inherit.
	It provides some common functionalities for all Agents in the model.
	"""

	dic_role = {}

	def __init__(self, postman, acolor='red', mcolor='white', uid=None, idd=None,
		verbose=True, log_file=None, env=None, **paras):
		if uid is None:
			# This is unique for objects having overlapping lives.
			self.uid = id(self) 
		else:
			self.uid = uid

		self.id = idd

		self.paras = paras

		# Communication system for Agent (LetterBox and regsitration with Postman)
		self.letterbox = LetterBox(postman)
		postman.register_agent(self)
		self.env = env
		self.letterbox.add_agent(self)

		# Print and log parameters
		self.acolor = acolor
		self.mcolor = mcolor
		self.verbose = verbose
		self.log_file = log_file

		# Module functions to be added to the Agent
		self.receive_module_functions = []

		# Attributes being added dynamically based on paras dictionary
		for k, v in paras.items():
			setattr(self, k, v)

		# Methods to be executed on preparation of simulation
		self.on_prepare = []
		# Methods to be executed on finalisation of simulation
		self.on_finalise = []
		self.on_init = []

	def apply_agent_modifications(self):
		"""
		Modifications of Roles in agent based on loaded modules
		"""
		# First apply on_init function, if exists
		mam = copy(self.module_agent_modif)
		
		on_inits = mam.get('on_init', [])
		for on_init in on_inits:
			on_init(self)

		if 'on_init' in mam.keys():
			del mam['on_init']

		if 'new_parameters' in mam.keys():
			del mam['new_parameters']

		for role_class, methods in mam.items():
			if type(methods) == dict:
				if role_class != 'on_init':
					role_name = self.dic_role[role_class]
					inst = getattr(self, role_name)
					for method_name, new_method in methods.items():
						if method_name == 'on_init':
							for nm in new_method:
								nm(inst)
						elif method_name == 'new':
							# New methods to attach
							for new_new_method in new_method:
								setattr(inst, new_new_method.__name__, new_new_method.__get__(inst))
						elif 'receive' in method_name:
							self.receive_module_functions.append(new_method)
						else:
							setattr(inst, method_name, new_method.__get__(inst))
			else:
				if role_class == 'on_prepare':
					# Methods to be executed on preparation of a simulation
					self.on_prepare.append(methods)
				elif role_class == 'on_finalise':
					# Methods to be excuted on finalisation of simulation
					self.on_finalise.append(methods)
				else:
					setattr(self, role_class, methods.__get__(self))

	def send(self, msg):
		return self.letterbox.send(msg)

	def receive(self, msg):
		pass

	def prepare_for_simulation(self):
		[m(self) for m in self.on_prepare]

	def finalise(self):
		[m(self) for m in self.on_finalise]


class Role:
	"""
	Base Role from which all Roles in Mercury inherit.
	It provides some common functionalities for all Roles in the model.
	"""

	def __init__(self, agent):
		"""
		On initialisation of a Role, it gets a pointer to the Agent it belongs to.
		"""
		self.agent = agent

	def send(self, msg):
		"""
		Function to send messages to its own agent
		"""
		# For convenience
		self.agent.send(msg)
