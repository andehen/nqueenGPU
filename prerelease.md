### 26th May: Pre-release 
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
