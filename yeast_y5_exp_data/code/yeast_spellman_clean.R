library(data.table)
library(zoo)

my_LOCF <- function(vec){
  if(is.na(vec[1])){
    first_non_blank_idx = which(!is.na(vec))[1]
    vec[1] <- vec[first_non_blank_idx] #NOCB
  }
  
  new_vec <- na.locf(vec, na.rm = FALSE) #LOCF
  
  return(new_vec)  
}

experiment_extractor <- function(this_data, exper){
  #spellman_id_genes <- fread("C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/spellman_id_genes.csv")
  spellman_id_genes <- fread("C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/all_spellman_orftogene.csv")
  spellman_id_genes <- spellman_id_genes[, .(ORF, SGD)]
  subset_data <- merge(this_data[experiment == exper], spellman_id_genes, 
                       by.x = "gene", by.y = "ORF" )
  full_blank_genes <-  subset_data[,all(is.na(value)),by = gene][V1 == TRUE, gene]
  subset_data <- subset_data[!gene %in% full_blank_genes][order(gene, time)]
 # subset_data <- subset_data[SGD %in% common_genes]
  subset_data[, 
                 expression_LOCF := my_LOCF(value),
                 by = .(SGD)]
  
  X = dcast(subset_data, 
        SGD + gene + experiment ~ time, 
        value.var = "expression_LOCF")
  
  #setnames(X, old = "SGD", new = "gene")
  
  time_cols <- setdiff(names(X), c("gene","SGD","experiment"))
  time_cols_new <- paste("time", time_cols, sep="_")
  setnames(X, old = time_cols, 
           new = time_cols_new)
  
  stage_times <- as.numeric(time_cols)
  X <- X[, .SD[1:(.N+1)],
         by=experiment][is.na(gene),
                        (time_cols_new) := as.list(stage_times)]
  
  gene_names <- X[!is.na(gene), .(orf = gene, sgd = SGD)]
  #SGD_names <- X[!is.na(gene), unique(SGD)]
  X[, c("gene","SGD","experiment") := NULL]
  
  top_row <- as.list(rep(NA, length(time_cols)))
  top_row[[1]] <- gene_names[,.N]
  top_row[[2]] <- 1
  
  X <- rbind(top_row, X)
  return(list(transformed_data = X,
              gene_sgd_names = gene_names))
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

cdc15_data_list <- experiment_extractor(full_data, "cdc15")
#cdc28_data_list <- experiment_extractor(full_data, "cdc28")
#alpha_data_list <- experiment_extractor(full_data, "alpha")
#elu_data_list <- experiment_extractor(full_data, "elu")


write.table( cdc15_data_list$transformed_data,
             "C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/clean_data/yeast_cdc15_5915genes_1sample_24T.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")
write.csv(cdc15_data_list$gene_sgd_names,
          "C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/cdc15_5915gene_names.csv",
          row.names = F)


write.table(cdc28_data_list$transformed_data,
             "C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/clean_data/yeast_cdc28_786genes_1sample_17T.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")
write.csv(cdc28_data_list$gene_names,
          "C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/cdc28_names.csv",
          row.names = F)

common_genes <- intersect(cdc15_data_list$gene_names,
                          cdc28_data_list$gene_names)

write.table(alpha_data_list$transformed_data,
            "C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/clean_data/yeast_alpha_787genes_1sample_18T.csv", 
            sep=",",
            row.names = FALSE,
            col.names = FALSE,
            na = "")
write.csv(alpha_data_list$gene_names,
          "C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/alpha_names.csv",
          row.names = F)

write.table(elu_data_list$transformed_data,
            "C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/clean_data/yeast_elu_787genes_1sample_14T.csv", 
            sep=",",
            row.names = FALSE,
            col.names = FALSE,
            na = "")
write.csv(elu_data_list$gene_names,
          "C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/elu_names.csv",
          row.names = F)

short_genes <- fread("C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/spellman_id_genes.csv")
long_genes <- fread("C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/all_spellman_orftogene.csv")

short_genes <- short_genes[, .(ORF, SGD)]
long_genes <- long_genes[, .(ORF, SGD)]
