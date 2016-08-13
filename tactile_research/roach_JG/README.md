RoACH
==========
This repository contains firmware and python control code for Josh's VelociRoACH robot
with the SkinProc board for the tactile shell.

This repository is part of the research project headed by Cem Koc, Can Koc, Brian Su in Biomimetic Millisystems Lab @ UC Berkeley.

Settings:
https://github.com/casarezc/imageproc-settings/tree/tactile

Main code:
https://github.com/casarezc/roach_JG/tree/tactile

Library: 27 April 2015 branch compiles ok. Later versions changed encoders, PID, radio, packets etc.
https://github.com/biomimetics/imageproc-lib/tree/c331ed52a4824d13d8d430589e5c245effd4c5d0

The python code to run the robot is:
python experiment_mulitbot.py

To plot data use: force_plotting_from_telem_example.m


Build status: [![Build Status](https://travis-ci.org/biomimetics/roach.svg?branch=master)](https://travis-ci.org/biomimetics/roach)  
Built against biomimetics/roach 'master branch:
https://github.com/biomimetics/roach


Files:
---------
 firmware/   -  contains the C code firmware for the robot, and the MPLAB IDE project files.
 lib/		 -  C code library of modules for the octoroach firmware
 python/ 	 -	python code for controlling the robot from a PC, and examples.
 doc/		 -  documentation on firmware and python code.

Instructions
-------------
To be updated
