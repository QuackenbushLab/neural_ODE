#----load scripts and libraries----
#load libraries
liblist = c('parallel', 'doSNOW', 'stringr','MASS',
            "nleqslv", "data.table", "pracma")
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

update_graph_taus <- function(grnThis, new_tau){
  grnThat <- copy(grnThis)
  nodes <- grnThat@nodeset
  for (node in names(nodes)){
    grnThat@nodeset[[node]]@tau = new_tau
  }
  return(grnThat)
}


load("full_network.RData")

#seed to make multiple simulation reproducible
originseed = 60897

#generate seeds for 1000 simulations
set.seed(originseed)
simseeds = sample.int(1E7, 1000)

#----simulation parameters----
#simulation parameters
nsamp = 3 #number of samples
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

tau_series = seq(0.1, 1.3, by = 0.1)
time_series_data = list()
for (this_tau in tau_series){
  tau_label = paste("tau", this_tau, sep = "_")
  print(tau_label)
  grnSmall <- update_graph_taus(grnSmall, new_tau = this_tau)
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
  names(datamat) <- paste(names(datamat),tau_label, sep = "_")
  datamat[,node := row.names(simu_list$emat)]
  time_series_data[[tau_label]] <-  datamat
  
}

fulldatamat <- Reduce(merge, time_series_data)
edgepropmat = get_edge_params(grnSmall@edgeset)
ode_system_function = getODEFunc_modified(grnSmall)
fulldatamat = merge(fulldatamat, ode_system_function, by = "node")

write.csv(fulldatamat, 
          "C:/STUDIES/RESEARCH/ODE_project/clean_data/simulated_expression_20200709.csv", 
          row.names = F)
write.csv(edgepropmat, 
          "C:/STUDIES/RESEARCH/ODE_project/clean_data/edge_properties.csv", 
          row.names = F)


