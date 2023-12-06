#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path
import os

print ('Removing previous generated rst files...')
dir_path = Path('source') / Path('generated')
try:
	shutil.rmtree(dir_path)
except FileNotFoundError:
	pass

print("Generating new documentation...")
os.system('sphinx-build -M html source build')
