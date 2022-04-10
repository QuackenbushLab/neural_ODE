library(data.table)
library(stringr)


full_data <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/Data_smooth_allgenes.csv")
gene_names <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/GeneNames.csv",
                    header = F)
names(gene_names) <- "gene"
full_data <- cbind(gene_names, full_data)

full_data <- full_data[!is.na(gene) & gene != "", ]

full_data <- full_data[,.SD[1], by = gene]

exprs_cols <- setdiff(names(full_data),"gene")
full_data_melt <- melt(full_data, 
                       id.vars = "gene",
                       measure.vars = exprs_cols,
                       variable.name = "time_point",
                       value.name = "exprs")

full_data_melt[, hist(exprs)]
num_genes <- 500
high_var_genes <- full_data_melt[,var(exprs), by = gene][order(-V1)][1:num_genes,gene]


full_data <- full_data[gene %in% high_var_genes]
full_data <- full_data[order(gene)]
genes_names_to_save <- full_data[, .(gene)]

full_data[, gene := NULL]
full_data <- full_data/15 #NORMALIZE to 0-1

time_vals <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/LatantTime.csv")

full_data <- rbind(full_data, time_vals)
top_row <- as.list(rep(NA, dim(time_vals)[2]))
top_row[[1]] <- num_genes
top_row[[2]] <- 1
full_data <- rbind(top_row, full_data)

write.csv(genes_names_to_save , 
          "C:/STUDIES/RESEARCH/neural_ODE/breast_cancer_data/clean_data/desmedt_gene_names_500.csv",
          row.names = F)

write.table(full_data,
             "C:/STUDIES/RESEARCH/neural_ODE/breast_cancer_data/clean_data/desmedt_500genes_1sample_186T.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")
