"""
NOTE: this file should be updated for version 3.
NOTE: this file was moved from root folder in version 3.
How to use:
to compile binaries:
python3 setup.py build_ext 
to create wheel:
python3 setup.py bdist_wheel

Note on install with pip in new environment, use:

cat requirements.txt | xargs -n 1 pip3 install

instead of:

pip3 install -r requirements.txt

IMPORTANT: each foler has to have an __init__.py file in order to be compiled!
"""

import sys
sys.path.insert(1,'..')

from setuptools import setup
from setuptools.extension import Extension

import subprocess

from pathlib import Path
import shutil

from Cython.Build import cythonize
from Cython.Distutils import build_ext

from Mercury.model_version import model_version

name = 'Mercury'

libraries = ['libs/Hotspot', 'libs/uow_tool_belt', 'libs/performance_tools']#, 'agents/modules']
stuff_to_ignore = ['__pycache__', '.ipynb_checkpoints', '.idea', '.git']

extensions = [Extension("agents.*", ["agents/*.py"]),
				Extension("commodities.*", ["agents/commodities/*.py"]),
				#Extension("modules.*", ["agents/modules/*.py"]),
				Extension("libs.*", ["libs/*.py"]),
				Extension("performance_tools.*", ["libs/performance_tools/*.py"]),
				Extension("Hotspot.*", ["libs/Hotspot/*.py"]),
				Extension("uow_tool_belt.*", ["libs/uow_tool_belt/*.py"]),
				]
list_folder = []
for library in libraries:
	for name_file in Path(library).rglob("*"):
		if name_file.is_dir():
			cond = True
			for stuff in stuff_to_ignore:
				cond = cond and (not stuff in str(name_file))

			# Check if there is at least one .py file
			cond2 = False
			for name in name_file.iterdir():
				if (".py" in str(name)) and (not ".pyc" in str(name)):
					cond2 = True
					break
					
			cond = cond and cond2
			if cond:
				#print (name_file)
				#if 'ModelStructure' in str(name_file):
				extensions.append(Extension(str(name_file).replace('/', '_') +".*", [str(name_file)+"/*.py"]))
				list_folder.append(name_file)


for extension in extensions:
	print (extension)

class MyBuildExt(build_ext):
	def run(self):
		build_ext.run(self)

		build_dir = Path(self.build_lib) / name
		root_dir = Path(__file__).parent
		target_dir = build_dir if not self.inplace else root_dir

		self.copy_file(Path('agents') / '__init__.py', root_dir, target_dir)
		self.copy_file(Path('agents/commodities') / '__init__.py', root_dir, target_dir)
		#self.copy_file(Path('agents/modules') / '__init__.py', root_dir, target_dir)
		self.copy_file(Path('libs') / '__init__.py', root_dir, target_dir)
		self.copy_file(Path('libs/performance_tools') / '__init__.py', root_dir, target_dir)
		self.copy_file(Path('libs/Hotspot') / '__init__.py', root_dir, target_dir)
		self.copy_file(Path('libs/uow_tool_belt') / '__init__.py', root_dir, target_dir)
		
		for fol in list_folder:
			if (fol / '__init__.py').exists():
				self.copy_file(fol / '__init__.py', root_dir, target_dir)
			else:
				print ("Can't find an __init__.py file in {}, I cannot create the corresponding binaries.".format(fol))


		self.copy_file('mercury.py', root_dir, target_dir)
		self.copy_file('model_version.py', root_dir, target_dir)
		self.copy_file('__init__.py', root_dir, target_dir)

		self.copy_file(Path('paras') / 'paras_scenario_template.py', root_dir, target_dir)
		self.copy_file(Path('paras') / 'paras_simulation_template.py', root_dir, target_dir)

		subprocess.run("pip3 freeze > requirements.txt", shell=True)
		
		self.copy_file('requirements.txt', root_dir, target_dir)
		self.copy_file('email_credentials_template.py', root_dir, target_dir)
		self.copy_file('credentials_template.py', root_dir, target_dir)
		self.copy_file('README.md', root_dir, target_dir)

	def copy_file(self, path, source_dir, destination_dir):
		if not (source_dir / path).exists():
			print (Path.cwd())
			print ("Can't find", source_dir / path)
			return

		(destination_dir / path).parent.mkdir(parents=True, exist_ok=True)

		shutil.copyfile(str(source_dir / path), str(destination_dir / path))

	def copy_folder(self, path, source_dir, destination_dir):
		if not (source_dir / path).exists():
			return
		if (destination_dir / path).exists():
			return
			
		shutil.copytree(str(source_dir / path), str(destination_dir / path))


setup(
	name=name,
	version=model_version,
	ext_modules=cythonize(extensions,
							build_dir="build",
							compiler_directives=dict(always_allow_keywords=True,
													language_level=3 # for Python 3
													)
							),
	cmdclass=dict(build_ext=MyBuildExt),
	#package_dir={'':'/home/earendil/Documents/Westminster/NOSTROMO/Model/Mercury'},
	packages=[],# ["agents", 'libs', "agents/commodities", "libs/performance_tools"]
	license='Exclusive',
	install_requires=['pandas', #TODO
						'numpy',
						]
	)