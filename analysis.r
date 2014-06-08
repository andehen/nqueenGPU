library(plyr)
library(reshape2)
library(ggplot2)
library(xtable)

#Set wd
setwd("/home/andehen/Dropbox/NTNU/4-kl/EPFL/Paralell Promgramming/nqueenGPU/")
theme_set(theme_bw())

TimePlot <- function(k){
	results.sub <- subset(results, K==k)
	ggplot(results.sub,aes(Searches,Time,colour=ARCH)) + geom_line() + geom_point() +
		ylab("Time [ms]") + xlab("Number of searches")
}

gridSizePlot <- function(k,s){
	results.sub <- subset(results, K == k & ARCH=="GPU" & Searches == s)
	ggplot(results.sub,aes(x=NUM_BLOCKS,y=Time)) + geom_bar(stat="identity") 
}

speedUpPlot <- function(k,n){
	results.gpu <- subset(results,K==k & ARCH=="GPU" & NUM_BLOCKS==n)
	results.cpu <- subset(results,K==k & ARCH=="CPU" & NUM_BLOCKS==n)
	results.merged <- merge(results.gpu,results.cpu,by="Searches")
	results.merged <- within(results.merged, {
					SpeedUp <- Time.y/Time.x 
		})
	ggplot(results.merged,aes(x=Searches,y=SpeedUp)) + geom_line()
}

printSpeedUpTable <- function(k,n){
	results.gpu <- subset(results,K==k & ARCH=="GPU" & NUM_BLOCKS==n)
	results.cpu <- subset(results,K==k & ARCH=="CPU" & NUM_BLOCKS==n)
	results.merged <- merge(results.gpu,results.cpu,by="Searches")
	results.merged <- within(results.merged, {
					SpeedUp <- Time.y/Time.x 
		})
	sptable <- results.merged[c("K.x","Searches","NUM_BLOCKS.x","NUM_THREADS.x","Time.x","Time.y","Solutions.x","Solutions.y","SpeedUp")]
	print(xtable(sptable),include.rownames=FALSE)
}

results <- read.table("results/results.csv", 
	col.names=c("ARCH","K","NUM_BLOCKS","NUM_THREADS","MAX_ITER","Time","Solutions"))

# Calculat num of searches
results <- within(results,{
		Searches <- NUM_BLOCKS*NUM_THREADS
	})
# Merge tables
sptable <- speedUpTable(40,16)
sptable <- sptable[c("K.x","Searches","NUM_BLOCKS.x","NUM_THREADS.x","Time.x","Time.y","Solutions.x","Solutions.y","SpeedUp")]

print(xtable(sptable),include.rownames=FALSE)