#----load scripts and libraries----
#load libraries
liblist = c('parallel', 'doSNOW')
#parliblist = c('nleqslv', 'foreach', 'MASS', 'EBcoexpress', 'plyr', 'qvalue',
#               'COSINE', 'nethet', 'stringr', 'reshape2', 'igraph', 'RColorBrewer',
#               'scales', 'PRROC', 'GeneNet')#, 'dcanr')
#sapply(c(liblist, parliblist), library, character.only = TRUE)

sapply(liblist, library, character.only = TRUE)

setwd("C:/STUDIES/RESEARCH/ODE_project/code/SimulatorGRN-master/")
#import simulator code
source('Classes.R')
source('GraphGRN_core.R')
source('GraphGRN.R')
source('SimulationGRN_core.R')
source('SimulationGRN.R')

#----build the network and overlay the mathematical model for the full network----
#parse the S. cerevisiae network into a GraphGRN object
net_srcfile = file.path('source_nets', 'Yeast_full.sif')
df = read.csv(net_srcfile, sep = '\t', header = FALSE, stringsAsFactors = FALSE)
df = df[df[, 2] %in% c('ac', 'du', 're'),]
edges = df
maplogical = c('ac' = TRUE, 're' = FALSE, 'du' = FALSE) # TRUE = activation, FALSE = repression
edges[,2] = maplogical[edges[,2]]

#seed to make multiple simulation reproducible
originseed = 60897

#read in network and randomize params
# no loops
# proportion of OR gates is 0
# randomisation seed used to generate networks in the paper
grnFull = df2GraphGRN(edges, loops = FALSE, propor = 0, seed = originseed)
#randomise model parameters and select linearlike activation functions
grnFull = randomizeParams(grnFull, 'linear-like', seed = originseed)
save(grnFull, file = "full_network.RData")


load("full_network.RData")

#generate seeds for 1000 simulations
set.seed(originseed)
simseeds = sample.int(1E7, 1000)

#----simulation parameters----
#simulation parameters
nsamp = 500 #number of samples
netSize = 150 #network size of sampled networks
minTFs = 10 #minimum number of TFs enforced on sampled networks
expnoise = 0.05 #experimental noise standard deviation (normal)
bionoise = 0.05 #biological noise standard deviation (superimposed log-normal)
propbimodal = 30 / netSize #proportion of bimodal genes (may be << prop*netSize)
ncores = 1 # number of processors to use (check avail using detectCores())

#use seed number 102 to perform one simulation
#using R3.5 will give the same results as sim102 packages in the dcanr package
# The random number generator was changed in R3.6 therefore different results
# will be generated
simseed = simseeds[102]

#create cluster
if (ncores > 1){
  cl = makeSOCKcluster(ncores, outfile = '')
  registerDoSNOW(cl)
  clusterCall(cl, function(libs) {
    sapply(libs, library, character.only = T)
    source('Classes.R')
    source('GraphGRN_core.R')
    source('GraphGRN.R')
    source('SimulationGRN_core.R')
    source('SimulationGRN.R')
    return()
  }, c(liblist, parliblist))
}

#----1: sample network and create simulation----
set.seed(simseed)
grnSmall = sampleGraph(grnFull, netSize, minTFs, seed = simseed)
grnSmall = randomizeParams(grnSmall, 'linear-like', simseed)
simSmall = new(
  'SimulationGRN',
  graph = grnSmall,
  seed = simseed,
  propBimodal = propbimodal,
  expnoise = expnoise,
  bionoise = bionoise
)

#----2: simulate data and generate truth matrix----
#identify genes with bimodal expression
bimgenes = unlist(lapply(simSmall$inputModels, function(x) length(x$prop)))
bimgenes = names(bimgenes)[bimgenes == 2]

#simulate dataset
datamat = simulateDataset(simSmall, nsamp)

#retrieve the condition matrix and check for bimodality in the data
condmat = attr(datamat, 'classf')
mincount = apply(apply(condmat, 1, table), 2, min)
#discard conditions that do not have enough observations in each group
if (all(mincount < 20)) {
  stop('Number of samples not enough to display conditional associations')
} else if (any(mincount < 20)) {
  warning(paste0('Some conditions removed due to low number of observations (',
                 paste(rownames(condmat)[mincount < 20], collapse = ', '), ')'))
  condmat = condmat[mincount >= 20, , drop = F]
  bimgenes = rownames(condmat)
}

#generate a truth adjacency matrix
inmodels = simSmall$inputModels #retrieve input model information
inputs = sapply(inmodels, function(x) x$mean[1]) #select means of models as inputs
#select mean of mixture with the highet mean for bimodal genes
inputs[bimgenes] = sapply(inmodels[bimgenes], function(x) x$mean[2])
names(inputs) = getInputNodes(grnSmall) #name input values with genes
#sensitivity analysis with 0.25 perturbations
sensmat = sensitivityAnalysis(simSmall, 0.25, unlist(inputs), names(inputs))
#generate a df of positive triplets with relative sensitivity thresholds of 0.7
triplets = getGoldStandard2(simSmall, sensmat = sensmat)
triplets = triplets[triplets$cond %in% bimgenes, ]
#retrieve list of genes co-regulated with modulators
condcoexmat = attr(triplets, 'cond')
condcoexmat = condcoexmat[bimgenes, , drop = F]

#----stop cluster if used----
if (ncores > 1)
  stopCluster(cl)


