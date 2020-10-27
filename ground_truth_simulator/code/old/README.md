A gene regulatory network simulator
================

This is the code used to simulate data from a gene regulatory network.
It can also be used to simulate data with differential associations.
Data in the `dcanr` R/Bioconductor package is generated using this
simulator in R 3.5. Note that the random number generator was changed in
R 3.6 therefore running these functions in R 3.6 will result in
different data sets despite the parameters and seeds being the same.

> example\_sim102.R shows an example simulation which was used to
> simulate the data in the `dcanr` package (`data(sim102)`)

## Documentation of code

Code in the package is organised into S4 classes. We created new S4
classes as existing graph-based classes were insufficient to add model
information. These classes can be extended to add more elements into the
model. Four classes have been developed in this code with additional
classes that inherit from them:

  - SimulationGRN: holds a GraphGRN object and additional simulation
    parameters
      - GraphGRN: holds a list of nodes and edges
          - Node: holds information for a single node (i.e. molecule)
              - NodeRNA: used to represent RNA molecules in the
                regulatory network
          - Edge: an association between two nodes
              - EdgeReg: an edge representing a regulatory
    relationship

## Method list

### SimulationGRN

#### Slots:

``` r
slotNames('SimulationGRN')
```

    ## [1] "graph"       "expnoise"    "bionoise"    "seed"        "inputModels"

  - graph - the regulatory network
  - expnoise - experimental noise standard deviation
  - bionoise - biological noise standard deviation
  - seed - randomisation seed
  - inputModels - means, variances and proportions defining the Gaussian
    mixture model for each input gene (auto generated)

#### Functions:

``` r
methods(class = 'SimulationGRN')
```

    ## [1] $                   $<-                 generateInputModels
    ## [4] initialize          sensitivityAnalysis show               
    ## [7] simulateDataset    
    ## see '?methods' for accessing help and source code

``` r
# simulate dataset from the simulation
showMethods('simulateDataset')
```

    ## Function: simulateDataset (package .GlobalEnv)
    ## simulation="SimulationGRN", numsamples="numeric", cor.strength="missing", externalInputs="ANY"
    ## simulation="SimulationGRN", numsamples="numeric", cor.strength="missing", externalInputs="missing"
    ## simulation="SimulationGRN", numsamples="numeric", cor.strength="numeric", externalInputs="missing"

  - numsamples - number of observations to simulate
  - cor.strength - strength of correlations between input genes (high
    value means stronger correlations 1-100)
  - externalInputs - specify input gene values manually (named matrix)

<!-- end list -->

``` r
sensitivityAnalysis(simulation, pertb, inputs, pertbNodes, tol)
```

  - pertb - amount of knockdown, 0.5 represents 50% reduction in
    abundance
  - input - input gene values
  - pertbNodes - nodes to perturb for computing the sensitivities
  - tol - tolerance for numerical errors

### GraphGRN

#### Slots:

``` r
slotNames('GraphGRN')
```

    ## [1] "nodeset" "edgeset"

  - nodeset - list of nodes
  - edgeset - list of
    edges

#### Functions:

``` r
methods(class = 'GraphGRN')
```

    ##  [1] addEdgeReg      addNodeRNA      edgenames       generateEqn    
    ##  [5] generateODE     getAM           getEdge         getEdge<-      
    ##  [9] getInputNodes   getNode         getNode<-       getSubGraph    
    ## [13] initialize      nodenames       randomizeParams removeEdge     
    ## [17] removeNode      sampleGraph     show           
    ## see '?methods' for accessing help and source code

``` r
# get names of edges and nodes
showMethods('nodenames')
```

    ## Function: nodenames (package .GlobalEnv)
    ## graph="GraphGRN"

``` r
showMethods('edgenames')
```

    ## Function: edgenames (package .GlobalEnv)
    ## graph="GraphGRN"

Regulation of a target node by multiple regulators A, B, and C results
in a pseudonode called *A\_B\_C* being created. Therefore such nodes
will appear in the edge and node names.

``` r
#get an adjacency representation of the graph
showMethods('getAM')
```

    ## Function: getAM (package .GlobalEnv)
    ## graph="GraphGRN", directed="logical"
    ## graph="GraphGRN", directed="missing"

  - directed - TRUE/FALSE determining whether to return a directed or
    undirected network

<!-- end list -->

``` r
# randomize activation functions for models
showMethods('randomizeParams')
```

    ## Function: randomizeParams (package .GlobalEnv)
    ## graph="GraphGRN", type="character", seed="numeric"

  - type - class of activation functions to use (`linear`,
    `linear-like`, `exponential`, `sigmoidal`, `mixed`)
  - seed - randomisation seed to use

<!-- end list -->

``` r
# sample a subgraph from the original network
showMethods('sampleGraph')
```

    ## Function: sampleGraph (package .GlobalEnv)
    ## graph="GraphGRN", size="numeric"

  - size - number of nodes to sample from the original network (i.e. new
    networks
size)

<!-- end list -->

``` r
# functions to switch beteen representations of the network, data.frame, igraph and GraphGRN.
df2GraphGRN(edges, nodes, propor, loops, seed)
GraphGRN2df(graph) # returns list of dfs, one for edges and one for nodes
GraphGRN2igraph(graph, directed)
```

  - edges - edge data frame as used in the `igraph` package, colnames of
    data frame can have colnames that match slot names of the
    `Edge`/`EdgeReg` class
  - nodes - node data frame as used in the `igraph` package, colnames of
    data frame can have colnames that match slot names of the
    `Node`/`NodeRNA` class
  - propor - proportion of `OR` gates to use when multiple regulators
    regulate a single target
  - loops - should loops be included in the model (ideally not)
  - seed - randomisation seed for parameter specification
  - directed - should the resulting network be
    directed

### Node/NodeRNA

#### Slots:

``` r
slotNames('Node')
```

    ## [1] "name"     "spmax"    "spdeg"    "inedges"  "outedges" "logiceqn"

``` r
slotNames('NodeRNA')
```

    ## [1] "tau"      "name"     "spmax"    "spdeg"    "inedges"  "outedges"
    ## [7] "logiceqn"

  - name - node name
  - spmax - maximum abundance of this molecule (from 0 - 1)
  - spdeg - degradation rate of this molecule (0 - 1)
  - tau - time constant (usually leave as 1 unless the aim is to
    simulate time-series data)
  - inedges - names of edges that feed in to this node (auto computed
    when added using `addEdge` function)
  - outedges - names of edges that go out of this node (auto computed
    when added using `addEdge` function)
  - logiceqn - the logic equation representing the regulation of this
    node e.g. A ^ B ^ C (auto
    generated)

#### Functions:

``` r
methods(class = 'NodeRNA')
```

    ## [1] $           $<-         addNodeRNA  generateEqn getNode<-   initialize 
    ## [7] show       
    ## see '?methods' for accessing help and source code

``` r
# add, retrieve, replace or remove a mRNA node
showMethods('addNodeRNA')
```

    ## Function: addNodeRNA (package .GlobalEnv)
    ## graph="GraphGRN", node="character", tau="ANY", spmax="ANY", spdeg="ANY", logiceqn="ANY", inedges="missing", outedges="missing"
    ## graph="GraphGRN", node="NodeRNA", tau="missing", spmax="missing", spdeg="missing", logiceqn="missing", inedges="missing", outedges="missing"

``` r
showMethods('getNode')
```

    ## Function: getNode (package .GlobalEnv)
    ## graph="GraphGRN", nodename="character"

``` r
showMethods('getNode<-')
```

    ## Function: getNode<- (package .GlobalEnv)
    ## graph="GraphGRN", nodename="character", value="Node"

``` r
showMethods('removeNode')
```

    ## Function: removeNode (package .GlobalEnv)
    ## graph="GraphGRN", nodenames="character"

  - nodenames - vector of node names to remove from the network
  - value - node to replace
    with

### Edge/EdgeReg

#### Slots:

``` r
slotNames('Edge')
```

    ## [1] "from"   "to"     "weight" "name"

``` r
slotNames('EdgeReg')
```

    ## [1] "EC50"       "n"          "activation" "from"       "to"        
    ## [6] "weight"     "name"

  - from - name of the source node
  - to - name of the target node
  - weight - weight of the interaction (from 0 - 1)
  - name - name of the edge (auto computed)
  - EC50 - conc. of regulator required for half maximal activation of
    the target
  - n - Hill constant for the activation function
  - activation - TRUE/FALSE representing activation/repression
    respectively

#### Functions:

``` r
methods(class = 'EdgeReg')
```

    ## [1] $                     $<-                   generateActivationEqn
    ## [4] getEdge<-             initialize            show                 
    ## see '?methods' for accessing help and source code

``` r
# add, retrieve, replace or remove a regulatory interaction
showMethods('addEdgeReg')
```

    ## Function: addEdgeReg (package .GlobalEnv)
    ## graph="GraphGRN", from="character", to="character"

``` r
showMethods('getEdge')
```

    ## Function: getEdge (package .GlobalEnv)
    ## graph="GraphGRN", from="character", to="character"

``` r
showMethods('getEdge<-')
```

    ## Function: getEdge<- (package .GlobalEnv)
    ## graph="GraphGRN", from="character", to="character", value="Edge"

``` r
showMethods('removeEdge')
```

    ## Function: removeEdge (package .GlobalEnv)
    ## graph="GraphGRN", from="character", to="character"
