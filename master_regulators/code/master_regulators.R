library(data.table)
library(ggplot2)
library(matrixStats)
#library(ggpubr)

run_sims = F

#chief_directory <- "/home/ubuntu/neural_ODE/master_regulators/"
chief_directory <- "C:/STUDIES/RESEARCH/neural_ODE/master_regulators"
write_directory <- paste(chief_directory,"score_outputs/scores_to_save_destination.csv", sep = "/")
img_directory <- paste(chief_directory,"plots/inflential_genes_destination.png", sep = "/")
img_directory_2 <- paste(chief_directory,"plots/central_metrics_destination.png", sep = "/")


times_to_project <- seq(0,10, by = 2)  
num_iter <- 60
pert_level <- 0.50

wo_prods <- fread(paste(chief_directory,"model_to_test/wo_prods.csv", sep = "/"))
bo_prods <- fread(paste(chief_directory,"model_to_test/bo_prods.csv", sep = "/"))
wo_sums <- fread(paste(chief_directory,"model_to_test/wo_sums.csv", sep = "/"))
bo_sums <- fread(paste(chief_directory,"model_to_test/bo_prods.csv", sep = "/"))
alpha_comb <- fread(paste(chief_directory,"model_to_test/alpha_comb.csv", sep = "/"))
gene_mult <- fread(paste(chief_directory,"model_to_test/gene_mult.csv", sep = "/"))
gene_names <- fread(paste(chief_directory,"model_to_test/gene_names.csv", sep = "/"))

num_genes <- dim(wo_prods)[1]
genes_in_dataset <- gene_names$x


true_edges <- fread(paste(chief_directory,"model_to_test/edge_properties.csv", sep = "/"))
true_edges[activation == T, activation_sym := "activating"]
true_edges[activation == F, activation_sym := "repressive"]
setnames(true_edges, 
         old = c("from","to"),
         new = c("reg","aff"))
true_outgoing <- true_edges[, .(true_out = .N), by = reg]
true_incoming <- true_edges[, .(true_inc = .N), by = aff]
true_nums <- merge(true_outgoing, true_incoming, 
                   by.x = "reg", by.y = "aff", all = T)
true_nums[is.na(true_out), true_out := 0 ]
true_nums[is.na(true_inc), true_inc := 0 ]

harmonic_cent <- fread(paste(chief_directory,"model_to_test/gene_centralities.csv", sep = "/"))
true_influences <- fread(paste(chief_directory,"model_to_test/true_influences.csv", sep = "/"))



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

  
score_matrix <- matrix(NA, nrow = num_genes, ncol = num_iter)
row.names(score_matrix) <- genes_in_dataset
colnames(score_matrix) <- paste("V",1:num_iter, sep = "")

if (run_sims == T){
  print("Running perturbations...")
  for (iter in 1:num_iter){
    baseline_init_val <- runif(num_genes, min = 0.1, max = 0.9)
    names(baseline_init_val) <- genes_in_dataset
    unpert_soln <- deSolve::ode(y = baseline_init_val, 
                                times = times_to_project, 
                        func = my_neural_ode)
    
    for (gene_counter in 1:num_genes){
      gene <- genes_in_dataset[gene_counter]
      print(paste("gene",gene_counter, "in iter", iter, "of", num_iter))
      pert_init_cond <- copy(baseline_init_val)
      this_gene_curr_val <- pert_init_cond[gene]
      pert_init_cond[gene] <- ifelse(runif(1) > 0.5, 
                                    this_gene_curr_val * (1 + pert_level), #perturb up
                                    this_gene_curr_val * (1 - pert_level)) #perturb down
      this_genes_pert_soln <- deSolve::ode(y = pert_init_cond, 
                                          times = times_to_project, 
                                          func = my_neural_ode)
      
      genes_to_consider <- setdiff(genes_in_dataset, gene) #ignore the perturbed gene itself
      delta_mat <- abs(this_genes_pert_soln[-1,genes_to_consider] - 
                                    unpert_soln[-1,genes_to_consider])
      times_considered <- c(2,4,6,8,10)
      #weights_considered <- (1*times_considered)^2
      weights_considered <- rep(1, length(times_considered))
      this_gene_score <- 10^4 * mean(diag(weights_considered) %*% delta_mat)
                                    #don't consider artifically perturbed init values (t = 0)!
      #print(this_gene_score)
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
score_summary <- data.table(gene = genes_in_dataset, pert_score = rowMedians(as.matrix(score_matrix)))
score_summary[,input_gene := grepl("_input",gene)]
score_summary[, gene_short:= gsub("_input","",gene)]
score_summary <- merge(score_summary, 
                       true_nums, 
                       by.x = "gene_short", 
                       by.y = "reg")
score_summary <-merge(score_summary,
                      harmonic_cent, 
                      by.x = "gene_short",
                      by.y = "gene")

 score_summary <- merge(score_summary,
                      true_influences, 
                      by.x = "gene_short",
                      by.y = "gene")                     


setnames(score_summary, old = c("out_cent","true_out","true_pert_median", "ivi_cent"),
                new = c("true_harmonic_cent", "true_out_degree", "true_influence", "true_ivi_cent"))

score_summary$true_out_degree <- as.numeric(score_summary$true_out_degree)

score_sd <- score_summary[, sd(pert_score)]
genes_to_print <- score_summary[pert_score > 2*score_sd,][order(-pert_score), gene]
print(score_summary[gene %in% genes_to_print,
                    .(gene, pert_score, 
                      true_harmonic_cent,
                   #   true_out_degree,
                      true_ivi_cent,
                      true_influence)][order(-pert_score), ])

#print("")
#print("Least influential genes:")
#print(score_summary[order(pert_score),
#                    .(gene, pert_score, 
#                      true_harmonic_cent,
                   #   true_out_degree,
#                      true_ivi_cent,
#                      true_influence)][1:5,])

print("")
print("influence breakdown by input gene:")
score_summary[,.(mean_score = mean(pert_score),
                 sd_score = sd(pert_score),
                # corr_out_degree =  cor(true_out_degree, pert_score, method = "pearson"),
                 corr_harmonic_cent = cor(true_harmonic_cent, pert_score, method = "pearson"),
                 corr_ivi_cent = cor(true_ivi_cent, pert_score, method = "pearson"),
                 corr_true_influence = cor(true_influence, pert_score, method = "pearson")), 
              by = input_gene]


print("")
print(paste("Making influence plots using all", num_iter, "iterations"))
boxplot_data <- score_matrix[genes_to_print,]
boxplot_data$gene <- row.names(boxplot_data)
boxplot_data <- data.table(boxplot_data)
boxplot_data <- melt(boxplot_data, id.vars = "gene",
                     measure.vars =  paste("V",1:num_iter, sep = ""),
                     variable.name = "sim_iter",
                     value.name = "perturb_score")

#boxplot_data[,perturb_score_norm := 
#               (perturb_score)/sd(perturb_score),
#             by = sim_iter]


png(filename=img_directory)
ggplot(boxplot_data, aes(x = factor(gene, 
                                    levels = genes_to_print),
                         y = perturb_score)) + 
  geom_boxplot(fill = "dodgerblue") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
  xlab("most influential genes")
dev.off()
  

scatterplot_data <- melt(score_summary, 
                          id.vars = c("gene","input_gene","pert_score"),
                          measure.vars = c("true_harmonic_cent","true_ivi_cent", "true_influence"),
                          variable.name = "true_metric", value.name = "true_metric_val")

png(filename=img_directory_2, height = 6, width = 6, units = "in", res = 1200)
ggplot(scatterplot_data, aes(x = pert_score, y = true_metric_val)) + 
  geom_point(col = "red") +
  #geom_smooth(method = "lm", se = FALSE, col = "black") + 
  theme_bw() + 
  facet_grid(factor(true_metric, 
                    levels = c("true_influence", "true_harmonic_cent", "true_ivi_cent"),
                    labels = c("Influence", "Harmonic centrality", "IVI (Salvaty 2020)")) ~., 
                     scales = "free") +
  xlab("Perturbation score from model") + 
  ylab("True metrics from ground truth GRN")
dev.off()

print("")
print("DONE!")


