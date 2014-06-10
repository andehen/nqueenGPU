Solving n-queen puzzle on GPU
=============================

This is a project in the course **Math-454 Parallel computing and POSIX Threads**
at EPFL. The aim of the project is to solve the n-queen puzzle on GPU by 
a state-space search approach. 

### 8th June: Project finished
See report for results and concusion. 

**Abstratct**

*This paper is about using GPU to find solutions to the n-queens puzzle. The puzzle was
solved by exploring the solution tree with random first search. This is a general constraint
satisfaction algorithm and can be applied to large number of problems. The general idea with using GPU to
this problems is to run multiple searches in parallel and thus find solutions faster. Benchmarking shows that compared to a CPU performing
the same number of searches sequentially, the GPU reduces the execution time by ~ 7.*