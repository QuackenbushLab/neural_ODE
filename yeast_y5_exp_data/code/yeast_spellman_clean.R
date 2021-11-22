library(data.table)

experiment_extractor <- function(this_data, exper){
  X = dcast(this_data[experiment == exper], 
        gene+experiment ~ time, 
        value.var = "value")
  
  time_cols <- setdiff(names(X), c("gene", "experiment"))
  setnames(X, old = time_cols, 
           new = paste("time", time_cols, sep="_"))
  return(X)
}

full_data <- read.delim("C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/combined.txt",
                        sep = "\t")
full_data <- data.table(full_data)
gene_names <- as.character(full_data[, X])
#nsamp <- 1 
setnames(full_data, old = "X", new = "gene")
full_data[, c("clb", "alpha", "cdc15", "cdc28", "elu"):= NULL]
exprs_col <- setdiff(names(full_data), "gene")
full_data <- melt(full_data, 
                  id.vars = c("gene"),
                  measure.vars = exprs_col)
full_data[grepl("cln3", variable), experiment := "cln3"]
full_data[grepl("clb2", variable), experiment := "clb2"]
full_data[grepl("alpha", variable), experiment := "alpha"]
full_data[grepl("cdc15", variable), experiment := "cdc15"]
full_data[grepl("cdc28", variable), experiment := "cdc28"]
full_data[grepl("elu", variable), experiment := "elu"]

full_data[, time:= gsub("cln3\\.|clb2\\.|alpha|cdc15_|cdc28_|elu","",variable)]
full_data[, time := as.numeric(time)]
full_data[, variable:= NULL]

full_data[, table(experiment)]

cdc15_data <- experiment_extractor(full_data, "cdc15")
cdc28_data <- experiment_extractor(full_data, "cdc28")

#y5_data <- fread("C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/y5_gene_exp.csv")
#common_genes <- cdc28_data[toupper(gene) %in% toupper(y5_data$Gene), as.character(gene)]
#plot(as.matrix(cdc28_data[toupper(gene) %in% common_genes, 3:19]),
#     as.matrix(y5_data[toupper(Gene) %in% common_genes, 3:19])
#     )
alpha_data <- experiment_extractor(full_data, "alpha")


time_cols <- c("c1","c2")
stage_times <- c(1,2)

datamat <- perturbed_inits[, .SD[1:(.N+1)],
                   by = .(pert_gene)]


datamat[is.na(c1),
        (time_cols) := as.list(stage_times)]

datamat[,c("pert_gene") := NULL]
top_row <- as.list(rep(NA, 2))
top_row[[1]] <- num_genes
top_row[[2]] <- num_genes

datamat <- rbind(top_row, datamat)

write.table( datamat,
             "C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/clean_data/yeast_perturbation_inits.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")
