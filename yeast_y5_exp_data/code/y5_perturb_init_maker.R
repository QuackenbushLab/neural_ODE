library(data.table)
library(ggplot2)


full_data <- fread("C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/y5_gene_exp.csv")
num_genes <- full_data[, length(unique(Gene))]
nsamp <- 1 


perturbed_inits <- data.table(pert_gene = numeric(), 
                              c1 = numeric(),
                              c2 = numeric())
init_cond <- full_data$c1

for(i in 1:num_genes){
  new_init_cond <- copy(init_cond)
  new_init_cond[i] <- -5
  perturbed_inits <- rbind(perturbed_inits, 
                               list(pert_gene = rep(i, num_genes),
                                    c1 = new_init_cond,
                                    c2 = 999))
}

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
