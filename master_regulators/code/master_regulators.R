library(data.table)
library(ggplot2)

chief_directory <- "C:/STUDIES/RESEARCH/neural_ODE/master_regulators/model_to_test"

wo_prods <- fread(paste(chief_directory,"wo_prods.csv", sep = "/"))
bo_prods <- fread(paste(chief_directory,"bo_prods.csv", sep = "/"))
wo_sums <- fread(paste(chief_directory,"wo_sums.csv", sep = "/"))
bo_sums <- fread(paste(chief_directory,"bo_prods.csv", sep = "/"))
alpha_comb <- fread(paste(chief_directory,"alpha_comb.csv", sep = "/"))
gene_mult <- fread(paste(chief_directory,"gene_mult.csv", sep = "/"))
gene_names <- fread(paste(chief_directory,"gene_names.csv", sep = "/"))

num_genes <- dim(wo_prods)[1]
genes_in_dataset <- gene_names$x

soft_sign_mod <- function(x){
  shift_x <- x - 0.5
  abs_shift_x <- abs(x)
  ss_mod <- shift_x/(1+ abs_shift_x)
  return(ss_mod)
}

log_soft_sign_mod <- function(x){
  shift_x <- x - 0.5
  abs_shift_x <- abs(x)
  ss_mod <- shift_x/(1+ abs_shift_x)
  return(log(1 + ss_mod))
}


my_neural_ode <- function(t, y, parms = NULL,...){
  y_soft_sign <- soft_sign_mod(y)
  y_log_soft_sign <- log_soft_sign_mod(y)
  sums_part <- t(y_soft_sign%*%as.matrix(wo_sums)) + bo_sums
  prods_part <- exp(t(y_log_soft_sign%*%as.matrix(wo_prods)) + bo_prods)
  concat <- rbind(sums_part, prods_part)
  joint = t(as.matrix(alpha_comb)) %*% as.matrix(concat)
  final <- gene_mult*(joint - y)
  return(list(final$V1))
}

  
times_to_project <- seq(0,10, by = 2)  
num_iter <- 1
pert_level <- 0.50
score_matrix <- matrix(NA, nrow = num_genes, ncol = num_iter)
row.names(score_matrix) <- genes_in_dataset


for (iter in 1:num_iter){
  baseline_init_val <- runif(num_genes, min = 0, max = 1)
  names(baseline_init_val) <- genes_in_dataset
  unpert_soln <- deSolve::ode(y = baseline_init_val, 
                              times = times_to_project, 
                       func = my_neural_ode)
  
  for (gene_counter in 1:num_genes){
    gene <- genes_in_dataset[gene_counter]
    print(paste("gene",gene_counter, "in iter", iter))
    pert_init_cond <- copy(baseline_init_val)
    pert_init_cond[gene] <- ifelse(runif(1) > 0.5, 
                                   pert_init_cond[gene] * (1 + pert_level),
                                   pert_init_cond[gene] * (1 - pert_level))
    this_genes_pert_soln <- deSolve::ode(y = pert_init_cond, 
                                         times = times_to_project, 
                                         func = my_neural_ode)
    
    
    this_gene_score <- mean(abs(this_genes_pert_soln[,genes_in_dataset] - 
                                  unpert_soln[,genes_in_dataset]))
    score_matrix[gene, iter] <- this_gene_score
  }
  
  
}

