library(data.table)

current_names <- fread("C:/STUDIES/RESEARCH/neural_ODE/pramila_yeast_data/pramila3551_gene_names.csv")
conv_file <- fread("C:/STUDIES/RESEARCH/neural_ODE/pramila_yeast_data/gProfiler_names.csv")

conv_file <- conv_file[, .(initial_alias, name)]

current_names <- merge(current_names, conv_file,
      by.x = "x", by.y = "initial_alias", 
      all.x = T)

write.csv(current_names,
          "C:/STUDIES/RESEARCH/neural_ODE/pramila_yeast_data/pramila3551_gene_names.csv",
          row.names = F)

