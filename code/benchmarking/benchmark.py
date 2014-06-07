from string import Template
import os
import subprocess
import numpy as np


def measureTime(k, num_blocks, num_threads, max_iter, results_file):
	
	parameters = {"NUM_BLOCKS":num_blocks, "NUM_THREADS":num_threads, "K": k, "MAX_ITER": max_iter}
	
	# Open files
	src_cuda = open("kqueens.cu","r")
	src_cpp = open("kqueens.cpp","r")
	
	# Create templates of source
	cuda_src_temp = Template(src_cuda.read())
	cpp_src_temp = Template(src_cpp.read())

	src_cuda.close()
	src_cpp.close()

	# Create tmp folder to store generated soruce files
	os.mkdir("tmp")
	os.chdir("tmp")

	## CUDA
	# Out file
	out_cuda = open("gpu.cu","w")
	out_cuda.write(cuda_src_temp.substitute(parameters))
	out_cuda.close()

	# Compile
	subprocess.call("nvcc gpu.cu -w -arch=sm_30 -o rungpu",shell=True)

	# Run
	res_cuda = np.array([0,0])
	for i in xrange(5):
		res_cuda += [float(x) for x in subprocess.check_output("./rungpu").split("\n")[:2]]
	res_cuda /= 5

	# Clean
	os.remove("rungpu")
	os.remove("gpu.cu")
	
	## CPU
	# Out file
	out_cpu = open("cpu.cpp","w")
	out_cpu.write(cpp_src_temp.substitute(parameters))
	out_cpu.close()

	# Compile
	subprocess.call("g++ cpu.cpp -o runcpu",shell=True)

	# Run
	res_cpu = np.array([0,0])
	for i in xrange(5):
		res_cpu += [float(x) for x in subprocess.check_output("./runcpu").split("\n")[:2]]
	res_cpu /= 5

	# Clean
	os.remove("cpu.cpp")
	os.remove("runcpu")	
	os.chdir("..")
	os.rmdir("tmp")

	# Write results
	results_file.write("GPU \t %s \t %s \t %s \t %s \t %s \t %s\n" % (k, num_blocks, num_threads, max_iter, res_cuda[0], res_cuda[1]))
	results_file.write("CPU \t %s \t %s \t %s \t %s \t %s \t %s\n" % (k, num_blocks, num_threads, max_iter, res_cpu[0], res_cpu[1]))


fresults = open("results.csv","a")
measureTime(20, 256, 256, 2000, fresults)
measureTime(40, 256, 256, 2000, fresults)
measureTime(60, 256, 256, 2000, fresults)
measureTime(80, 256, 256, 2000, fresults)
measureTime(100,256, 256, 2000, fresults)


