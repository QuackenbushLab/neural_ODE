library(data.table)
library(stringr)

#first do it for the training data

full_data <-fread("C:/STUDIES/RESEARCH/neural_ODE/pramila_yeast_data/clean_data/pramila_3551genes_1sample_24T.csv")
time_row <- full_data[.N, ]
full_data <- full_data[-c(1, .N),]

full_data[, rowVar := apply(.SD, 1, var), .SDcols = names(full_data)]
full_data[, gene_SL := .I]

high_var_genes <- sort(full_data[, .(gene_SL, rowVar),][order(-rowVar)][1:500, gene_SL])

full_data[, rowVar := NULL]
full_data[, gene_SL := NULL]
full_data <- full_data[high_var_genes, ]

top_row <- as.list(rep(NA, length(time_row)))
top_row[[1]] <- 500
top_row[[2]] <- 1

full_data <- rbind(top_row, full_data, time_row)
write.table( full_data,
             "C:/STUDIES/RESEARCH/neural_ODE/pramila_yeast_data/clean_data/pramila_500genes_1sample_24T.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")

#Now for testing data
full_data <-fread("C:/STUDIES/RESEARCH/neural_ODE/pramila_yeast_data/clean_data/pramila_3551genes_1VALsample_24T.csv")
time_row <- full_data[.N, ]
full_data <- full_data[-c(1, .N),]
full_data <- full_data[high_var_genes, ]

top_row <- as.list(rep(NA, length(time_row)))
top_row[[1]] <- 500
top_row[[2]] <- 1

full_data <- rbind(top_row, full_data, time_row)
write.table( full_data,
             "C:/STUDIES/RESEARCH/neural_ODE/pramila_yeast_data/clean_data/pramila_500genes_1VALsample_24T.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")

#Now create subsetted prior
full_prior <- as.matrix(fread("C:/STUDIES/RESEARCH/neural_ODE/pramila_yeast_data/clean_data/edge_prior_matrix_pramila_3551.csv"))
full_prior <- full_prior[high_var_genes, ]
full_prior <- full_prior[, high_var_genes]
write.table( full_data,
             "C:/STUDIES/RESEARCH/neural_ODE/pramila_yeast_data/clean_data/edge_prior_matrix_pramila_500.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")

#Now store gene names 
full_names <- fread("C:/STUDIES/RESEARCH/neural_ODE/pramila_yeast_data/pramila3551_gene_names.csv")
full_names <- full_names[high_var_genes, ]
write.csv(full_names, 
          "C:/STUDIES/RESEARCH/neural_ODE/pramila_yeast_data/pramila500_gene_names.csv", 
          row.names = F)
