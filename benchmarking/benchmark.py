from utils import measureTime

# Example usage

# File to append results
fresults = open("results.csv","a")

 # Input: K, num_blocks, num_threads, max_iter, results_file
measureTime(20, 16, 128, 2000, fresults)
measureTime(40, 32, 64, 2000, fresults)
measureTime(100, 16, 128, 2000, fresults)