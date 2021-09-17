library(data.table)
library(ggplot2)
library(matrixStats)

source("/home/ubuntu/neural_ODE/master_regulators/code/my_true_ode.R")

chief_directory <- "/home/ubuntu/neural_ODE/master_regulators/"
write_directory <- paste(chief_directory,"model_to_test/true_influences4.csv", sep = "/")


times_to_project <- seq(0,10, by = 2)  
num_iter <- 15
pert_level <- 0.50

gene_names <- fread(paste(chief_directory,"model_to_test/gene_names.csv", sep = "/"))
genes_in_dataset <- gene_names$x
num_genes <- length(genes_in_dataset)

input_genes <- grep("_input", genes_in_dataset, value = TRUE)
non_input_genes <- setdiff(genes_in_dataset, input_genes)
input_genes <- gsub("_input","",input_genes)
genes_in_dataset <- gsub("_input","",genes_in_dataset)

score_matrix <- matrix(NA, nrow = num_genes, ncol = num_iter)
row.names(score_matrix) <- genes_in_dataset
colnames(score_matrix) <- paste("V",1:num_iter, sep = "")

print("Running perturbations...")
for (iter in 1:num_iter){
baseline_init_val <- runif(num_genes, min = 0.1, max = 0.9)
names(baseline_init_val) <- genes_in_dataset

unpert_soln <- deSolve::ode(y = baseline_init_val[non_input_genes], 
                            times = times_to_project, 
                            func = my_true_ode,
                            parms = baseline_init_val[input_genes])

input_gene_vals <- matrix( baseline_init_val[input_genes], 
                            nrow = length(times_to_project),
                            ncol = length(input_genes),
                            byrow = T)

colnames(input_gene_vals) <- input_genes   
unpert_soln <- cbind(unpert_soln, input_gene_vals)                         

for (gene_counter in 1:num_genes){
    gene <- genes_in_dataset[gene_counter]
    print(paste("gene",gene_counter, "in iter", iter, "of", num_iter))
    pert_init_cond <- copy(baseline_init_val)
    this_gene_curr_val <- pert_init_cond[gene]
    pert_init_cond[gene] <- ifelse(runif(1) > 0.5, 
                                this_gene_curr_val * (1 + pert_level), #perturb up
                                this_gene_curr_val * (1 - pert_level)) #perturb down

    this_genes_pert_soln <- deSolve::ode(y = pert_init_cond[non_input_genes], 
                                        times = times_to_project, 
                                        func = my_true_ode,
                            parms = pert_init_cond[input_genes])
    
    curr_input_gene_vals <- matrix( pert_init_cond[input_genes], 
                            nrow = length(times_to_project),
                            ncol = length(input_genes),
                            byrow = T)

    colnames(curr_input_gene_vals) <- input_genes  
    this_genes_pert_soln <- cbind(this_genes_pert_soln, curr_input_gene_vals)    
    genes_to_consider <- setdiff(genes_in_dataset, gene) #ignore the perturbed gene itself

    this_gene_score <- 10^4 * mean(abs(this_genes_pert_soln[-1,genes_to_consider] - 
                                unpert_soln[-1,genes_to_consider]))
                                #don't consider artifically perturbed init values (t = 0)!
    score_matrix[gene, iter] <- this_gene_score 
}

}

#score_matrix <- data.table(score_matrix)
print("saving results now...")
write.csv(score_matrix, write_directory, row.names = T)

score_summary <- data.table(gene = genes_in_dataset, pert_score = rowMedians(as.matrix(score_matrix)))
print(score_summary[order(-pert_score), ][1:10])



