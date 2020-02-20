Instruction to use the code package.

Requirements:
1. Python. The code was tested in versions: 3.7.3 & 3.7.4
2. PyTorch. The code was tested in versions: 1.1.0 & 1.2.0
3. Numpy version 1.16.4
4. Scipy version 1.3.0
5. Matplotlib version 3.1.0
6. ipython (optional) version 7.6.1


Notes: 
1. The USPS file is given with the package.
2. The data file is a pickle object which is zipped. No need to unzip as the code will take care of it.
2. 'CEVisualizer.py' requires command line argument.
3. The script runs with default dataset 'USPS' when no argument is passed.

How to run the code:
1. Download the code.
2. Make sure all the requirements are satisfied.
3. Run the script with default mode: From command like:
	python CEVisualizer.py
4. Run the script with argument: From command like:
	python CEVisualizer.py MNIST
5. To run the script from ipython us the commands:
	a. run CEVisualizer.py <= default mode
	b. run CEVisualizer.py MNIST <= with argument

