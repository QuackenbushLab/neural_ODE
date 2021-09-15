library(data.table)
library(ggplot2)

run_sims = F

#chief_directory <- "/home/ubuntu/neural_ODE/master_regulators/"
chief_directory <- "C:/STUDIES/RESEARCH/neural_ODE/master_regulators"
write_directory <- paste(chief_directory,"score_outputs/scores_to_save.csv", sep = "/")
img_directory <- paste(chief_directory,"plots/inflential_genes.png", sep = "/")



wo_prods <- fread(paste(chief_directory,"model_to_test/wo_prods.csv", sep = "/"))
bo_prods <- fread(paste(chief_directory,"model_to_test/bo_prods.csv", sep = "/"))
wo_sums <- fread(paste(chief_directory,"model_to_test/wo_sums.csv", sep = "/"))
bo_sums <- fread(paste(chief_directory,"model_to_test/bo_prods.csv", sep = "/"))
alpha_comb <- fread(paste(chief_directory,"model_to_test/alpha_comb.csv", sep = "/"))
gene_mult <- fread(paste(chief_directory,"model_to_test/gene_mult.csv", sep = "/"))
gene_names <- fread(paste(chief_directory,"model_to_test/gene_names.csv", sep = "/"))

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
num_iter <- 10
pert_level <- 0.50
score_matrix <- matrix(NA, nrow = num_genes, ncol = num_iter)
row.names(score_matrix) <- genes_in_dataset

if (run_sims == T){
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
      
      
      this_gene_score <- 10^4 * mean(abs(this_genes_pert_soln[,genes_in_dataset] - 
                                    unpert_soln[,genes_in_dataset]))
      score_matrix[gene, iter] <- this_gene_score
    }
    
    
  }
  
  #score_matrix <- data.table(score_matrix)
  print("saving results now...")
  write.csv(score_matrix, write_directory, row.names = T)

}else{
  score_matrix <-  read.csv(write_directory)
  row.names(score_matrix) <- score_matrix$X
  score_matrix <- score_matrix[,-1]
  
}

print("")
print("top-most influential genes (2 SDs away) are:")
score_summary <- data.table(gene = genes_in_dataset,pert_score = rowMeans(score_matrix))
score_sd <- score_summary[, sd(pert_score)]
genes_to_print <- score_summary[pert_score > 2*score_sd,][order(-pert_score), gene]
print(score_summary[gene %in% genes_to_print,][order(-pert_score), ])

print("")
print("influence breakdown by input gene:")
score_summary[,input_gene := grepl("_input",gene)]
score_summary[,.(mean_score = mean(pert_score),
                 sd_score = sd(pert_score)), 
              by = input_gene]


print("")
print("Making influence plots")
boxplot_data <- score_matrix[genes_to_print,]
boxplot_data$gene <- row.names(boxplot_data)
boxplot_data <- data.table(boxplot_data)
boxplot_data <- melt(boxplot_data, id.vars = "gene",
                     measure.vars =  paste("V",1:num_iter, sep = ""),
                     variable.name = "sim_iter",
                     value.name = "perturb_score")


png(filename=img_directory)
ggplot(boxplot_data, aes(x = factor(gene, 
                                    levels = genes_to_print),
                         y = perturb_score)) + 
  geom_boxplot(fill = "dodgerblue") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
  xlab("most influential genes")
dev.off()
  
  
