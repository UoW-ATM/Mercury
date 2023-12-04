#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

cmd = 'cd .. && pdoc Mercury -o Mercury/doc --logo "https://github.com/UoW-ATM/Mercury/blob/documentation/mercury_logo_small.png"'

subprocess.run(cmd, shell=True)
