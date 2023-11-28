#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NOTE: NOT COMPATIBLE WITH WINDOWS. TODO!!!!
Note: after executing this script, one can use docker like this:
docker build -t mercury .

And then run it like this:

docker run -p 8888:8888 mercury
"""

import sys
sys.path.insert(1,'..')

import argparse
from pathlib import Path

import shutil

from Mercury.libs.uow_tool_belt.connection_tools import read_data, write_data, generic_connection
from Mercury.libs.uow_tool_belt.general_tools import read_paras

def get_everything(path_save,
	path_binaries='build/lib.linux-x86_64-3.8/Mercury',
	path_input=None,
	keep_non_zip=True):
	
	path_s = Path(path_save)

	# root path
	path_r = Path(__file__).parent.absolute()
	
	# Remove previous shipment
	try:
		shutil.rmtree(path_s)
	except FileNotFoundError:
		pass

	path_s.parent.mkdir(parents=True,
						exist_ok=True)
	
	# Copy binaries (from binary folder)
	path_b = Path(path_binaries)

	shutil.copytree(path_b,
					path_s / 'Mercury',
					dirs_exist_ok=True)

	# Copy input data (from input folder)
	path_i = Path(path_input)

	shutil.copytree(path_i,
					path_s / "input",
					dirs_exist_ok=True)

	(path_s / "Mercury/paras").mkdir(parents=True,
						exist_ok=True)

	# Copy other things directly from source code folder
	# Copy local credentials
	src = 'local_dtu_credentials.py'
	dst = path_s / 'Mercury/local_credentials.py'
	shutil.copyfile(src, dst)

	# Copy email credentials
	src = 'email_credentials_template.py'
	dst = path_s / 'Mercury/email_credentials_template.py'
	shutil.copyfile(src, dst)

	# Copy email credentials
	src = 'README.md'
	dst = path_s / 'Mercury/README.md'
	shutil.copyfile(src, dst)

	# Copy requirements
	src = 'requirements.txt'
	dst = path_s / 'Mercury/requirements.txt'
	shutil.copyfile(src, dst)

	# Copy paramter files
	src = 'paras/my_paras_scenario_dtu_nostromo.py'
	dst = path_s / 'Mercury/paras/my_paras_scenario.py'
	shutil.copyfile(src, dst)

	src = 'paras/my_paras_simulation_dtu_nostromo.py'
	dst = path_s / 'Mercury/paras/my_paras_simulation.py'
	shutil.copyfile(src, dst)

	src = 'paras/paras_scenario_template.py'
	dst = path_s / 'Mercury/paras/paras_scenario_template.py'
	shutil.copyfile(src, dst)

	src = 'paras/paras_simulation_template.py'
	dst = path_s / 'Mercury/paras/paras_simulation_template.py'
	shutil.copyfile(src, dst)

	# Copy notebook demonstration
	src = 'Interactive Mercury session.ipynb'
	dst = path_s / 'Mercury' / 'Interactive Mercury session.ipynb'
	shutil.copyfile(src, dst)

	# Copy main script
	src = 'mercury.py'
	dst = path_s / 'Mercury' / 'mercury.py'
	shutil.copyfile(src, dst)

	# Copy init
	src = '__init__.py'
	dst = path_s / 'Mercury' / '__init__.py'
	shutil.copyfile(src, dst)

	# Copy model version
	src = 'model_version.py'
	dst = path_s / 'Mercury' / 'model_version.py'
	shutil.copyfile(src, dst)

	# Copy a file from hotspot that does not work properly when compiled.
	src = 'libs/Hotspot/UDPP/LocalOptimised/udppLocalOptGuroby.py'
	dst = path_s / 'Mercury/libs/Hotspot/UDPP/LocalOptimised' / 'udppLocalOptGuroby.py'
	shutil.copyfile(src, dst)
	src = path_s / 'Mercury/libs/Hotspot/UDPP/LocalOptimised' / 'udppLocalOptGuroby.cpython-38-x86_64-linux-gnu.so'
	dst = path_s / 'Mercury/libs/Hotspot/UDPP/LocalOptimised' / 'udppLocalOptGuroby.cpython-38-x86_64-linux-gnu.so.old'
	shutil.move(src, dst)

	# Copy a file from hotspot that does not work properly when compiled.
	src = 'libs/Hotspot/GlobalOptimum/SolversGO/gurobi_solver_go.py'
	dst = path_s / 'Mercury/libs/Hotspot/GlobalOptimum/SolversGO' / 'gurobi_solver_go.py'
	shutil.copyfile(src, dst)
	src = path_s / 'Mercury/libs/Hotspot/GlobalOptimum/SolversGO' / 'gurobi_solver_go.cpython-38-x86_64-linux-gnu.so'
	dst = path_s / 'Mercury/libs/Hotspot/GlobalOptimum/SolversGO' / 'gurobi_solver_go.cpython-38-x86_64-linux-gnu.so.old'
	shutil.move(src, dst)

	# Copy a file from hotspot that does not work properly when compiled.
	src = 'libs/Hotspot/NNBound/SolversNNB/gurobi_solver_NNB.py'
	dst = path_s / 'Mercury/libs/Hotspot/NNBound/SolversNNB' / 'gurobi_solver_NNB.py'
	shutil.copyfile(src, dst)
	src = path_s / 'Mercury/libs/Hotspot/NNBound/SolversNNB' / 'gurobi_solver_NNB.cpython-38-x86_64-linux-gnu.so'
	dst = path_s / 'Mercury/libs/Hotspot/NNBound/SolversNNB' / 'gurobi_solver_NNB.cpython-38-x86_64-linux-gnu.so.old'
	shutil.move(src, dst)

	# Copy module files in clear because of module discovery broken when compiled
	src = 'agents/modules'
	dst = path_s / 'Mercury/agents/modules'
	shutil.copytree(src, dst, dirs_exist_ok=True)

	# Remove all compiled files in modules
	src = 'agents/modules'
	dst = path_s / 'Mercury/agents/modules'
	shutil.copytree(src, dst, dirs_exist_ok=True)

	for f in Path(path_s / 'Mercury/agents/modules').glob('**/*'):
		if f.suffix=='.so':
			f.unlink()

	# # TODO: FIX THAT
	# src = path_r / 'agents' / 'modules' / 'FAC_FIFO_queue.py'
	# dst = path_s / 'Mercury' / 'agents' / 'modules' / 'FAC_FIFO_queue.py'
	# shutil.copyfile(src, dst)

	# Copy stuff for Hotspot
	coin = path_r / 'libs' / 'Hotspot' / 'ModelStructure' / 'Costs'
	for file in coin.iterdir():
		name = file.name
		if file.is_file() and name[:-3]!='.py':
			src = coin / name
			dst = path_s / 'Mercury' / 'libs' / 'Hotspot' / 'ModelStructure' / 'Costs' / name
			shutil.copyfile(src, dst)

	coin = path_r / 'libs' / 'Hotspot' / 'OfferChecker'
	for file in coin.iterdir():
		name = file.name
		if file.is_file() and name[:-3]!='.py':
			src = coin / name
			dst = path_s / 'Mercury' / 'libs' / 'Hotspot' / 'OfferChecker' / name
			shutil.copyfile(src, dst)

	# Make zip
	shutil.make_archive(str(path_s), 'zip', path_s)

	print ("Binaries copied in", path_save, ' along with zip file.')

	if not keep_non_zip:
		try:
			shutil.rmtree(path_s)
		except FileNotFoundError:
			pass


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Ship Mercury for binaries',
									add_help=True)
	
	parser.add_argument('-psi',
						'--paras_simulation',
						help='parameter file for simulation',
						required=False,
						default='paras/my_paras_simulation.py',
						nargs='?')

	# TODO: add the input we want to be shipped
	# parser.add_argument('-id',
	# 					'--id_scenario',
	# 					help='parameter file for simulation',
	# 					required=False,
	# 					default=None,
	# 					nargs='*')

	args = parser.parse_args()

	paras_simulation = read_paras(paras_file=args.paras_simulation)
	
	get_everything('build/binaries',
					path_input=paras_simulation['read_profile']['path'],
					keep_non_zip=True)