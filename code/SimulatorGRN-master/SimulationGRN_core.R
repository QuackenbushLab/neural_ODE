#----Simulation: functions----
validSimulationGRN <- function(object) {
	#graph is a GraphGRN object
	if (!is(object@graph, 'GraphGRN')) {
		stop('graph must be a GraphGRN object')
	}
	
	#local noise
	if (object@expnoise<0) {
		stop('Ecperimental noise standard deviation must be greater than 0')
	}
	
	#global noise
	if (object@bionoise<0) {
		stop('Biological noise standard deviation must be greater than 0')
	}
	
	return(TRUE)
}

initSimulationGRN <- function(.Object, ..., graph, expnoise = 0, bionoise = 0, seed = sample.int(1e6,1), inputModels = list(), propBimodal = 0) {
	.Object@graph = graph
	.Object@expnoise = expnoise
	.Object@bionoise = bionoise
	.Object@seed = seed
	.Object@inputModels = inputModels
	
	if (length(inputModels) == 0) {
	  .Object = generateInputModels(.Object, propBimodal)
	}
	
	validObject(.Object)
	return(.Object)
}

solveSteadyState <- function(object, externalInputs) {
  #external inputs
  if (is.null(names(externalInputs)) |
      !all(names(externalInputs) %in% getInputNodes(object@graph))) {
    stop('Invalid external inputs vector, named vector expected for ALL input nodes')
  }
  
  #set random seed
  set.seed(object@seed)
  
  #solve ODE
	ode = generateODE(object@graph)
	ext = externalInputs
	graph = object@graph
	nodes = setdiff(nodenames(graph), names(ext))
	exprs = rbeta(length(nodes), 2, 2)
	exprs[exprs < 0] = 0
	exprs[exprs > 1] = 1
	names(exprs) = nodes
	
	soln = nleqslv(exprs, ode, jac = NULL, ext)
	
	#check if convergence is reached or not
	if(soln$termcd != 1) {
	  warning('Solution not achieved. use \'diagnostics(simulation)\' to get details')
	}
	return(soln)
}

createInputModels <- function(simulation, propBimodal) {
  set.seed(simulation@seed)
  
  #create input models
  innodes = getInputNodes(simulation@graph)
  inmodels = list()
  
  for (n in innodes) {
    parms = list()
    mxs = sample(c(1, 2), 1, prob = c(1 - propBimodal, propBimodal))
    
    if (mxs == 2) {
      parms = c(parms, 'prop' = runif(1, 0.2, 0.8))
      parms$prop = c(parms$prop, 1 - parms$prop)
      parms$mean = c(rbeta(1, 10, 100), rbeta(1, 10, 10))
    } else {
      parms$prop = 1
      parms$mean = rbeta(1, 10, 10)
    }
    
    maxsd = pmin(parms$mean, 1 - parms$mean) / 3
    parms$sd = sapply(maxsd, function(x) max(rbeta(1, 15, 15) * x, 0.01))
    inmodels = c(inmodels, list(parms))
  }
  
  names(inmodels) = innodes
  simulation@inputModels = inmodels
  
  return(simulation)
}

#src = https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices
#src = Lewandowski, Kurowicka, and Joe (LKJ), 2009
#lower betaparam gives higher correlations
vineS <- function(d, betaparam = 5, seed = sample.int(1E6, 1)) {
  set.seed(seed)
  P = matrix(rep(0, d ^ 2), ncol = d)
  S = diag(rep(1, d))
  
  for (k in 2:(d - 1)) {
    for (i in (k + 1):d) {
      P[k, i] = rbeta(1, betaparam, betaparam)
      P[k, i] = (P[k, i] - 0.5) * 2
      p = P[k, i]
      for (l in (k - 1):1) {
        p = p * sqrt((1 - P[l, i] ^ 2) * (1 - P[l, k] ^ 2)) + P[l, i] * P[l, k]
      }
      S[k, i] = p
      S[i, k] = p
    }
  }
  
  permutation = sample(1:d, d)
  S = S[permutation, permutation]
  
  return(S)
}

# beta = 0 means no correlated inputs, smaller beta means stronger correlations
generateInputData <- function(simulation, numsamples, cor.strength = 5) {
  set.seed(simulation@seed)
  
  innodes = getInputNodes(simulation@graph)
  externalInputs = matrix(-1,nrow = numsamples, ncol = length(innodes))
  colnames(externalInputs) = innodes
  
  #create input models
  if (length(simulation@inputModels) == 0) {
    simulation = generateInputModels(simulation)
  }
  
  #simulate external inputs
  inmodels = simulation@inputModels
  classf = c()
  for (n in innodes) {
    m = inmodels[[n]]
    mix = sample(1:length(m$prop), numsamples, prob = m$prop, replace = T)
    
    outbounds = 1
    while (sum(outbounds) > 0){
      outbounds = externalInputs[ , n] < 0 | externalInputs[ , n] > 1
      externalInputs[outbounds & mix == 1, n] = rnorm(sum(outbounds & mix == 1), m$mean[1], m$sd[1])
      if (length(m$prop) > 1) {
        externalInputs[outbounds & mix == 2, n] = rnorm(sum(outbounds & mix == 2), m$mean[2], m$sd[2])
      }
    }
    
    if (length(m$prop) > 1) {
      #save class information
      classf = rbind(classf, mix)
      rownames(classf)[nrow(classf)] = n
    }
  }
  
  #correlated inputs
  if (cor.strength > 0 & numsamples > 1) {
    inputs = ncol(externalInputs)
    dm = apply(externalInputs, 2, sort)
    covmat = vineS(inputs, cor.strength, simulation@seed)
    cordata = mvrnorm(numsamples, rep(0, inputs), covmat)
    for (i in 1:inputs) {
      #avoid correlated bimodal inputs
      if (i %in% which(innodes %in% rownames(classf))) {
        cordata[, i] = externalInputs[, i]
      } else {
        cordata[, i] = dm[, i][rank(cordata[, i])]
      }
    }
    
    externalInputs = cordata
  }
  
  #add mixture info to attributes
  attr(externalInputs, 'classf') = classf
  
  colnames(externalInputs) = innodes
  return(externalInputs)
}

#cor.strength used for generating correlated inputs
simDataset <- function(simulation, numsamples, cor.strength, externalInputs) {
  #browser()
  if (missing(cor.strength)) {
    cor.strength = 5
  }
  
  #generate input matrix
  innodes = getInputNodes(simulation@graph)
  if (!missing(externalInputs) && !is.null(externalInputs)) {
    if (nrow(externalInputs) != numsamples |
        length(setdiff(innodes, colnames(externalInputs))) != 0) {
          stop('Invalid externalInputs matrix provided')
    }
    externalInputs = externalInputs[, innodes, drop = F]
    classf = NULL
  } else{
    externalInputs = generateInputData(simulation, numsamples, cor.strength)
    
    #extract class information
    classf = attr(externalInputs, 'classf')
  }
  
  #set random seed
  set.seed(simulation@seed)
  
  #solve ODE
  graph = simulation@graph
  ode = generateODE(graph)
  
  #generate LN noise for simulation
  lnnoise = exp(rnorm(numsamples * length(nodenames(graph)), 0, simulation@bionoise))
  lnnoise = matrix(lnnoise, nrow = numsamples, byrow = T)
  colnames(lnnoise) = nodenames(graph)
  
  #initialize solutions
  nodes = setdiff(nodenames(graph), colnames(externalInputs))
  exprs = rbeta(length(nodes) * numsamples, 2, 2)
  exprs[exprs < 0] = 0
  exprs[exprs > 1] = 1
  exprs = matrix(exprs, nrow = numsamples)
  colnames(exprs) = nodes

  #solve ODEs for different inputs
  res = foreach(i = 1:numsamples, .packages = c('nleqslv'), .combine = cbind) %dopar% {
    print(paste("solving",i,"of",numsamples))
    soln = nleqslv(exprs[i, ], ode, externalInputs = externalInputs[i, ], lnnoise = lnnoise[i, ])
    return(c(soln$x, soln$termcd))
  }
  
  res = cbind(res)
  
  termcd = res[nrow(res),]
  emat = res[-(nrow(res)), , drop = F]
  emat = rbind(emat, t(externalInputs))
  colnames(emat) = paste0('sample_', 1:numsamples)

  #check for errors
  if (!all(termcd == 1)) {
    nc = termcd != 1
    msg = 'Simulations for the following samples did not converge:'
    sampleids = paste(colnames(emat)[nc], ' (', termcd[nc], ')', sep = '')
    msg = paste(c(msg, sampleids), collapse = '\n\t')
    msg = paste(msg, 'format: sampleid (termination condition)', sep = '\n\n\t')
    warning(msg)

    emat = emat[, !nc]
  }
  
  #add experimental noise
  expnoise = rnorm(nrow(emat) * ncol(emat), 0, simulation@expnoise)
  expnoise = matrix(expnoise, nrow = nrow(emat), byrow = T)
  emat = emat + expnoise
  
  #add class information to attributes
  if (!is.null(classf)) {
    classf = classf[, termcd == 1, drop = F]
    colnames(classf) = colnames(emat)
    attr(emat, 'classf') = classf
  }
  
  return(list(emat = emat, lnnoise = lnnoise))
}

generateSensMat <- function(simulation, pertb, inputs = NULL, pertbNodes = NULL, tol = 1E-3) {
  set.seed(simulation@seed)
  graph = simulation@graph
  
  if (pertb < 0 | pertb > 1) {
    stop('Perturbation (knock-down) should be between 0 and 1.')
  }
  
  if (is.null(inputs)) {
    inputs = runif(length(getInputNodes(graph)), pertb + 1E-4, 1)
    names(inputs) = getInputNodes(graph)
  }else if (!all(getInputNodes(graph) %in% names(inputs))) {
    stop('Missing Inputs')
  }
  
  if (is.null(pertbNodes)) {
    pertbNodes = nodenames(graph)
  } else{
    pertbNodes = intersect(pertbNodes, nodenames(graph))
  }
  
  #generate ODE functions
  graph = simulation@graph
  ode = generateODE(graph)
  
  #generate LN noise for simulation - 0 noise
  lnnoise = exp(rep(0, length(nodenames(graph))))
  names(lnnoise) = nodenames(graph)
  
  #initialize solutions
  nodes = setdiff(nodenames(graph), names(inputs))
  exprs = rbeta(length(nodes) * (1 + length(pertbNodes)), 2, 2)
  exprs[exprs < 0] = 0
  exprs[exprs > 1] = 1
  exprs = matrix(exprs, nrow = (1 + length(pertbNodes)))
  colnames(exprs) = nodes
  
  #original solution with no perturbations
  soln0 = nleqslv(exprs[(1 + length(pertbNodes)), ], ode, externalInputs = inputs, lnnoise = lnnoise)
  termcd0 = soln0$termcd
  soln0 = soln0$x
  soln0 = c(inputs, soln0)
  
  #solve ODEs for different perturbations
  res = foreach(i = 1:length(pertbNodes), .combine = rbind) %do% {
    n = pertbNodes[i]
    tempinputs = inputs
    if (n %in% names(inputs)){
      odefn = ode
      tempinputs[n] = max(tempinputs[n] * (1 - pertb), 1E-4)
    } else{
      getNode(graph, n)$spmax = getNode(graph, n)$spmax - pertb
      odefn = getODEFunc(graph)
      getNode(graph, n)$spmax = getNode(graph, n)$spmax + pertb
    }
    
    soln = nleqslv(exprs[i, ], odefn, externalInputs = tempinputs, lnnoise = lnnoise)
    tcd = soln$termcd
    soln = soln$x
    soln = c(tempinputs, soln, tcd)
    return(soln)
  }
  res = rbind(c(), res) #if numeric vector returned, covert to matrix
  termcds = res[, ncol(res), drop = F]
  res = res[, -ncol(res), drop = F]
  res = rbind(res)
  
  sensmat = c()
  for (i in 1:length(pertbNodes)) {
    n = pertbNodes[i]
    
    #calculate sensitivity
    diffexpr = (res[i, ] - soln0)
    diffexpr[abs(diffexpr) < tol] = 0 #small difference resulting from numerical inaccuracies
    sensmat = rbind(sensmat, diffexpr/-pertb * getNode(graph, n)$spmax/soln0)
  }
  
  #if base case does not converge, no sensitivity analysis possible
  if (termcd0 != 1) {
    warning('Convergance not achieved for unperturbed case')
    sensmat = sensmat * 0
  }
  
  #if some solutions do not converge, set all their sensitivities to 0
  termcds = termcds == 1
  if (!all(termcds)) {
    warning('Convergance not achieved for SOME perturbations')
    termcds = as.numeric(termcds)
    termcds = termcds %*% t(rep(1, ncol(sensmat)))
    sensmat = sensmat * termcds
  }
  
  rownames(sensmat) = pertbNodes
  
  #since sensitivity calculation depends on the solver, round sensitivities to 
  #account for such numerical inaccuracies
  sensmat = round(sensmat, digits = round(-log10(tol)))
  
  return(sensmat)
}

#only derives the truth when the bimodal genes are input nodes
getGoldStandard <- function(simulation, threshold = 0.7, assocnet = T, sensmat = NULL) {
  #extract variables from the model
  graph = simulation@graph
  
  #get bimodal genes
  bimodal = unlist(lapply(simulation$inputModels, function(x) length(x$prop)))
  bimodal = names(bimodal)[bimodal == 2]
  if (length(bimodal) == 0) {
    stop('No conditional associations in the network')
  }
  
  #perform sensitivity analysis on the model if required
  inputs = sapply(simulation$inputModels, function(x) x$mean[1])
  inputs[bimodal] = 0.5
  names(inputs) = getInputNodes(graph)
  if (is.null(sensmat)) {
    sensmat = sensitivityAnalysis(simulation, 0.25, inputs, nodenames(graph))
  } else if (!all(rownames(sensmat) %in% colnames(sensmat)) |
             !all(colnames(sensmat) %in% rownames(sensmat))) {
    stop('Sensitivity matrix must be square and with all genes perturbed')
  }
  
  sensmat = sensmat[, rownames(sensmat)]
  diag(sensmat) = 0
  
  #generate condcoex mat
  condcoexmat = sensmat[bimodal, , drop = F] * 0
  
  innodes = names(inputs)
  triplets = c()
  for (b in bimodal) {
    #generate a normalized matrix with input node sensitivities
    inmat = sensmat[innodes, setdiff(colnames(sensmat), innodes)]
    inmat = abs(inmat) / matrix(1, nrow = nrow(inmat)) %*% colSums(abs(inmat))
    inmat[is.nan(inmat)] = 0
    #identify direct targets and conditionally regulated targets
    condcoex = c(colnames(inmat)[inmat[b, ] >= threshold], b)
    coregtgts = colnames(inmat)[inmat[b, ] > 0 & inmat[b, ] < threshold]
    condcoexmat[b, condcoex] = 1
    #identify conditionally dependent pairs
    diffpairs = sensmat[, coregtgts, drop = F] * matrix(1, nrow = nrow(sensmat)) %*% sensmat[b, coregtgts]
    diffpairs = sqrt(abs(diffpairs)) * sign(diffpairs)
    diffpairs[condcoex, ] = 0
    diffpairs = melt(diffpairs)
    diffpairs = diffpairs[diffpairs$value != 0, ]
    colnames(diffpairs) = c('TF', 'Target', 'strength')
    diffpairs$TF = as.character(diffpairs$TF)
    diffpairs$Target = as.character(diffpairs$Target)
    if (nrow(diffpairs) == 0)
      next
    diffpairs$inferred = F
    
    if (assocnet) {
      #sibling effect
      intfs = intersect(unique(diffpairs$TF), innodes)
      tfpairs = inmat[intfs, , drop = F]
      tfpairs[sensmat[intfs, colnames(tfpairs)] < 1] = 0
      
      #other downstream TFs
      for (t in setdiff(unique(diffpairs$TF), intfs)) {
        #discard upstream TF from normalization step
        tfmat = sensmat[sensmat[, t] == 0, setdiff(colnames(sensmat), innodes)]
        tfmat = abs(tfmat) / matrix(1, nrow = nrow(tfmat)) %*% colSums(abs(tfmat))
        tfmat = tfmat[t, , drop = F]
        tfmat[t, abs(sensmat[t, colnames(tfmat)]) < 1] = 0 #sensitivity thresholdold
        tfpairs = rbind(tfpairs, tfmat)
      }
      
      #select downstream targets that may be highly correlated
      tfpairs[abs(tfpairs) < 1] = 0
      tfpairs = melt(tfpairs)
      tfpairs = tfpairs[tfpairs$value != 0 & ! is.na(tfpairs$value), ]
      tfpairs[, 1] = as.character(tfpairs[, 1])
      tfpairs[, 2] = as.character(tfpairs[, 2])
      colnames(tfpairs)[1:2] = c('TF', 'newTF')
      
      if (nrow(tfpairs) != 0) {
        #add TFs to conditionally regulated pairs list
        tfpairs = merge(diffpairs, tfpairs, by = 'TF')
        tfpairs$strength = tfpairs$strength * tfpairs$value
        tfpairs = tfpairs[, c(5, 2:4)]
        colnames(tfpairs)[1] = 'TF'
        tfpairs$inferred = T
        #remove duplicates
        tfpairs = tfpairs[order(abs(tfpairs$strength), decreasing = T), ]
        tfpairs = tfpairs[!duplicated(tfpairs[, 1:2]), ]
        diffpairs = rbind(diffpairs, tfpairs)
      }
    }
    
    diffpairs = diffpairs[diffpairs$TF != diffpairs$Target, ]
    
    triplets = rbind(triplets, cbind('cond' = b, diffpairs))
  }
  
  #restructure triplets dataframe
  triplets = triplets[order(abs(triplets$strength), decreasing = T), ]
  rownames(triplets) = NULL
  # triplets[,2:3] = t(apply(triplets[,2:3],1,sort))
  # colnames(triplets)[1:3] = c('cond', 'x', 'y')
  triplets$known = T
  triplets$strength = -triplets$strength #positive correlation with z-scores
  
  #export condcoexmat as attribute
  attr(triplets, 'condcoex') = condcoexmat
  
  return(triplets)
}

#only derives the truth when the bimodal genes are input nodes
getGoldStandard2 <- function(simulation, sensmat = NULL) {
  #extract variables from the model
  graph = simulation@graph
  
  #get bimodal genes
  bimodal = unlist(lapply(simulation$inputModels, function(x) length(x$prop)))
  bimodal = names(bimodal)[bimodal == 2]
  if (length(bimodal) == 0) {
    stop('No conditional associations in the network')
  }
  
  #perform sensitivity analysis on the model if required
  inputs = sapply(simulation$inputModels, function(x) x$mean[1])
  inputs[bimodal] = 0.5
  names(inputs) = getInputNodes(graph)
  if (is.null(sensmat)) {
    sensmat = sensitivityAnalysis(simulation, 0.25, inputs, nodenames(graph))
  } else if (!all(rownames(sensmat) %in% names(inputs)) |
             !all(colnames(sensmat) %in% nodenames(graph))) {
    stop('Sensitivity matrix must be square and with all genes perturbed')
  }
  
  sensmat[cbind(rownames(sensmat), rownames(sensmat))] = 0
  sensmat = abs(sensmat) > 0.01
  
  #generate condcoex mat
  condcoexmat = sensmat[bimodal, , drop = F] * 0
  triplets = c()
  for (b in bimodal) {
    if (sum(sensmat[b, ]) == 0)
      next
    
    #identify direct targets and conditionally regulated targets
    bmat = sensmat[, sensmat[b, ], drop = F]
    bmat = bmat[rowSums(bmat) != 0, , drop = F]
    condcoex = colnames(bmat)[bmat[b, ] & colSums(bmat) == 1]
    coregtgts = colnames(bmat)[bmat[b, ] & colSums(bmat) > 1]
    condcoexmat[b, condcoex] = 1
    
    #identify conditionally dependent pairs
    bmat = bmat[!rownames(bmat) %in% b, coregtgts, drop = F]
    diffpairs = melt(bmat)
    diffpairs = diffpairs[diffpairs$value, 1:2]
    colnames(diffpairs) = c('TF', 'Target')
    diffpairs$TF = as.character(diffpairs$TF)
    diffpairs$Target = as.character(diffpairs$Target)
    if (nrow(diffpairs) == 0)
      next
    
    #select downstream genes for coregulating inputs
    diffpairs = ddply(diffpairs, 'Target', function(x) {
      newtfs = colnames(sensmat)[colSums(sensmat[x$TF, , drop = F]) == colSums(sensmat) &
                                   colSums(sensmat) != 0]
      diffdf = data.frame('TF' = c(x$TF, newtfs), stringsAsFactors = F)
      return(diffdf)
    })
    
    diffpairs = diffpairs[diffpairs$TF != diffpairs$Target, ]
    triplets = rbind(triplets, cbind('cond' = b, diffpairs[, 2:1]))
  }
  
  #restructure triplets dataframe
  rownames(triplets) = NULL
  triplets$known = T
  
  #distances
  nodedist = distances(GraphGRN2igraph(graph), mode = 'out')
  nodedist = melt(nodedist)
  names(nodedist) = c('TF', 'Target', 'Dist')
  triplets = merge(triplets, nodedist, all.x = T)
  triplets = triplets[, c(3, 1:2, 4:ncol(triplets))]
  triplets$Direct = triplets$Dist==1
  triplets$Influence = !is.infinite(triplets$Dist)
  triplets$Association = T
  
  #export condcoexmat as attribute
  attr(triplets, 'condcoex') = condcoexmat
  
  return(triplets)
}
