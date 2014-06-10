library(plyr)
library(reshape2)
library(ggplot2)
library(xtable)

# Some functions used to generate plots and tables from results/results.csv

theme_set(theme_bw())

# Plot execution time as function of seraches perfomed
TimePlot <- function(k){
	results.sub <- subset(results, K==k)
	ggplot(results.sub,aes(Searches,Time,colour=ARCH)) + geom_line() + geom_point() +
		ylab("Time [ms]") + xlab("Number of searches")
}

# Plot execution time as function of grid size parameters
gridSizePlot <- function(k,s){
	results.sub <- subset(results, K == k & ARCH=="GPU" & Searches == s)
	ggplot(results.sub,aes(x=NUM_BLOCKS,y=Time)) + geom_bar(stat="identity") 
}

# Plot speedup as function of searches
speedUpPlot <- function(k,n){
	results.gpu <- subset(results,K==k & ARCH=="GPU" & NUM_BLOCKS==n)
	results.cpu <- subset(results,K==k & ARCH=="CPU" & NUM_BLOCKS==n)
	results.merged <- merge(results.gpu,results.cpu,by="Searches")
	results.merged <- within(results.merged, {
					SpeedUp <- Time.y/Time.x 
		})
	ggplot(results.merged,aes(x=Searches,y=SpeedUp)) + geom_line()
}

# Print result table with all interesting details
speedUpTable <- function(k,n){
	results.gpu <- subset(results,K==k & ARCH=="GPU" & NUM_BLOCKS==n)
	results.cpu <- subset(results,K==k & ARCH=="CPU" & NUM_BLOCKS==n)
	results.merged <- merge(results.gpu,results.cpu,by="Searches")
	results.merged <- within(results.merged, {
					SpeedUp <- Time.y/Time.x 
		})
	results.merged[c("K.x","Searches","NUM_BLOCKS.x","NUM_THREADS.x",
							"Time.x","Time.y","Solutions.x","Solutions.y","SpeedUp")]
}

# Read in results
results <- read.table("results/results.csv", 
	col.names=c("ARCH","K","NUM_BLOCKS","NUM_THREADS","MAX_ITER","Time","Solutions"))

# Calculate num of searches
results <- within(results,{
		Searches <- NUM_BLOCKS*NUM_THREADS
	})

# Create data frame with results and speedup
sptable_list <- list(speedUpTable(40,16),speedUpTable(40,32),speedUpTable(40,64),
	speedUpTable(100,16),speedUpTable(100,64))

# Merge tables to one
merged_sptables <- Reduce(function(x,y) merge(x,y,all=TRUE),sptable_list)

# Print in latex format
print(xtable(merged_sptables),include.rownames=FALSE)