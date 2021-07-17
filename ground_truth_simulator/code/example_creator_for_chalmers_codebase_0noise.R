#----load scripts and libraries----
#load libraries
liblist = c('parallel', 'doSNOW', 'stringr','MASS',
            "nleqslv", "data.table", "deSolve")
sapply(liblist, library, character.only = TRUE)

setwd("C:/STUDIES/RESEARCH/neural_ODE/ground_truth_simulator/code")
#import simulator code
source('Classes.R')
source('GraphGRN_core.R')
source('GraphGRN.R')
source('SimulationGRN_core_init_var.R')
source('SimulationGRN_init_var.R')

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
originseed = 608978

#generate seeds for 1000 simulations
set.seed(originseed)
simseeds = sample.int(1E7, 1000)

#----simulation parameters----
#simulation parameters
nsamp = 150 #number of samples
netSize = 350 #network size of sampled networks
minTFs = 15 #minimum number of TFs enforced on sampled networks
expnoise = 0 #experimental noise standard deviation (normal)
bionoise = 0 #biological noise standard deviation (superimposed log-normal)
propbimodal = 0 #proportion of bimodal genes (may be << prop*netSize)

#use seed number 102 to perform one simulation
#using R3.5 will give the same results as sim102 packages in the dcanr package
# The random number generator was changed in R3.6 therefore different results
# will be generated
simseed = simseeds[102]

#----1: sample network and create simulation----
set.seed(simseed)
grnSmall = sampleGraph(grnFull, netSize, minTFs, seed = simseed)
#grnSmall = grnFull
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
time_stamps <- c(0,2,3,7,9) #0:9
simu_list = simulateDataset(simSmall, nsamp, 
                            timeStamps = time_stamps,
                            cor.strength = 0,
                            inputGeneVar  = 1,
                            outputGeneVar = 1) #1
datamat = simu_list$emat
edgepropmat = get_edge_params(grnSmall@edgeset)
ode_system_function = getODEFunc_modified(grnSmall)


cols_to_transpose <- setdiff(names(datamat),c("sample","time"))
time_cols <- paste("time",time_stamps,sep = "_")
datamat <- melt(datamat, id.vars = c("sample","time"),
     measure.vars = cols_to_transpose)
datamat <- dcast(datamat, sample + variable ~ time, value.var = "value")
names(datamat) <- c("sample","variable",time_cols)
datamat <- datamat[, .SD[1:(.N+1)], 
        by=sample][is.na(variable), (time_cols) := as.list(time_stamps)]
datamat[,c("sample","variable") := NULL]
top_row <- as.list(rep(NA, length(time_cols)))
top_row[[1]] <- netSize
top_row[[2]] <- nsamp

datamat <- rbind(top_row, datamat)


write.table( datamat,
             "C:/STUDIES/RESEARCH/neural_ODE/ground_truth_simulator/clean_data/chalmers_350genes_150samples_earlyT_0bimod_1initvar_nrange.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")

write.csv(edgepropmat, 
          "C:/STUDIES/RESEARCH/neural_ODE/ground_truth_simulator/clean_data/edge_properties.csv", 
          row.names = F)

write.csv(ode_system_function, 
          "C:/STUDIES/RESEARCH/neural_ODE/ground_truth_simulator/clean_data/ode_system_functions.csv", 
          row.names = F)



