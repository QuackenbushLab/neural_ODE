library(data.table)
library(stringr)


full_data <- read.delim("C:/STUDIES/RESEARCH/neural_ODE/pramila_yeast_data/YeastCCData_Expression.txt",
                        sep = "\t", header = F)

full_data <- data.table(full_data)
spellman_id_genes <- fread("C:/STUDIES/RESEARCH/neural_ODE/yeast_y5_exp_data/spellman_id_genes.csv")
spellman_id_genes <- spellman_id_genes$ORF

full_data[V1 %in% spellman_id_genes]
