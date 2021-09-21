library(data.table)
library(ggplot2)
library(matrixStats)
#library(ggpubr)

run_sims = F

#chief_directory <- "/home/ubuntu/neural_ODE/master_regulators/"
chief_directory <- "C:/STUDIES/RESEARCH/neural_ODE/hindcasting/"
img_directory <- paste(chief_directory,"plots/inflential_genes_destination.png", sep = "/")
img_directory_2 <- paste(chief_directory,"plots/central_metrics_destination.png", sep = "/")


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

for(i in 1:15){
  baseline_init_val <- runif(num_genes, min = 0.1, max = 0.9)
  names(baseline_init_val) <- genes_in_dataset
  soln_step_by_step <- deSolve::ode(y = baseline_init_val, 
                                    times = 0:10, 
                                    func = my_neural_ode)
  final_steps <- soln_step_by_step[11,2:351]
  
  soln_one_step <- deSolve::ode(y = baseline_init_val, 
                                times = c(0,10), 
                                func = my_neural_ode)
  
  final_one_step <- soln_one_step[2,2:351]
  
  print(paste("iter", i, 
              "cor is", cor(final_one_step, final_steps)))
  
}  

#OK so predicting step by step vs predicting in one go 
# doesn't make a difference (GOOD consistency!)

  target_val <- runif(num_genes, min = 0.1, max = 0.9)
  #target_val <- final_one_step
  names(target_val) <- genes_in_dataset
  soln_one_step <- deSolve::ode(y = target_val, 
                                times = c(1,0), 
                                func = my_neural_ode,
                                method = "bdf") #DOES NOT RUN!
  
  
  
  init_one_step <- soln_one_step[2,2:351]
  #FIX IT:
  init_one_step <- ifelse(init_one_step < 0, 0.001, init_one_step)
  init_one_step <- ifelse(init_one_step > 1, 1-0.001, init_one_step)
  
  final_using_init <- (deSolve::ode(y = init_one_step, 
                                   times = c(0,1), 
                                   func = my_neural_ode))[2,2:351]
  
  
  plot(target_val, final_using_init)
  


# IN general, hindcasting doesn't seem to be a feasible goal, 
# as not all target values will have corresponding inits! 
# a better strategy would be to train a separate network with 
# modified target-init pairs (modified trajectories), and see what
# the NN learns! 

