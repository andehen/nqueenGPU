Solving n-queen puzzle on GPU
=============================

This is a project in the course **Math-454 Parallel computing and POSIX Threads**
at EPFL. The aim of the project is to solve the n-queen puzzle on GPU by 
a state-space search approach. 

# Project finished
Project is finished. See the report for results and concusion.

On average I obtaind a speedup of ~ 7 compared to a single CPU. This can 
possibly be improved by making use of share memory rather than global
memory. 


### Pre-release 26th May
Here is a short status on the project. Report is far from finished. I have
focused most on the coding. It works, but I have some issues to figure out,
before I can obtain some good results. 

#### Issues
 - GPU device hangs when n > ca 50 
 - Memory leakage?
 - Threads in same block tend to generate same (not random) numbers

#### Todo
  - Find a way to terminate threads when one thread has found solution
  - Better memory management on GPU?
  - Find a better way to generate random numbers on GPU. 
