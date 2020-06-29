#----load scripts and libraries----
#load libraries
liblist = c('parallel', 'doSNOW', 'stringr','MASS',"nleqslv", "data.table")
sapply(liblist, library, character.only = TRUE)

setwd("C:/STUDIES/RESEARCH/ODE_project/code/SimulatorGRN-master/")
#import simulator code
source('Classes.R')
source('GraphGRN_core.R')
source('GraphGRN.R')
source('SimulationGRN_core.R')
source('SimulationGRN.R')

get_edge_params <- function(edgeset){
  X = lapply(edgeset,
             function(edge){
               return(list(from = edge@from,
                           to = edge@to,
                           weight = edge@weight,
                           activation = edge@activation,
                           EC50 = edge@EC50, 
                           n = edge@n))
               
             }
  )
  return(rbindlist(X))
}



load("full_network.RData")

#seed to make multiple simulation reproducible
originseed = 60897

#generate seeds for 1000 simulations
set.seed(originseed)
simseeds = sample.int(1E7, 1000)

#----simulation parameters----
#simulation parameters
nsamp = 10 #number of samples
netSize = 150 #network size of sampled networks
minTFs = 10 #minimum number of TFs enforced on sampled networks
expnoise = 0.05 #experimental noise standard deviation (normal)
bionoise = 0.05 #biological noise standard deviation (superimposed log-normal)
propbimodal = 30 / netSize #proportion of bimodal genes (may be << prop*netSize)

#use seed number 102 to perform one simulation
#using R3.5 will give the same results as sim102 packages in the dcanr package
# The random number generator was changed in R3.6 therefore different results
# will be generated
simseed = simseeds[102]

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
simu_list = simulateDataset(simSmall, nsamp)
datamat = data.table(simu_list$emat)
datamat[,node := row.names(simu_list$emat)]
edgepropmat = get_edge_params(grnSmall@edgeset)
ode_system_function = getODEFunc_modified(grnSmall)
datamat = merge(datamat, ode_system_function, by = "node")

write.csv(datamat, 
          "C:/STUDIES/RESEARCH/ODE_project/clean_data/simulated_expression.csv", 
          row.names = F)
write.csv(edgepropmat, 
          "C:/STUDIES/RESEARCH/ODE_project/clean_data/edge_properties.csv", 
          row.names = F)
