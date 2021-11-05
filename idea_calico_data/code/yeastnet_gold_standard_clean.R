library(data.table)

D = read.table("C:/STUDIES/RESEARCH/neural_ODE/idea_calico_data/YeastNet.v3.benchmark.txt")
D = data.table(D)

names(D) <- c("from", "to")

translate_dict <- fread("C:/STUDIES/RESEARCH/neural_ODE/idea_calico_data/orftogene.csv")
translate_dict[, desc := NULL]

D <- merge(D, translate_dict, by.x = "from", by.y = "orf_name")
D[, from := standard_name]
D[, standard_name := NULL]


D <- merge(D, translate_dict, by.x = "to", by.y = "orf_name")
D[, to := standard_name]
D[, standard_name := NULL]


all_genes <- fread("C:/STUDIES/RESEARCH/neural_ODE/idea_calico_data/gene_names_6175.csv")

prob_genes <- unique(c(unique(D$from[!D$from %in% all_genes$x]), unique(D$to[!D$to %in% all_genes$x])))


D[, edge_in_calico := to %in% all_genes$x & from %in% all_genes$x, 
  by = .(to, from)]

D[, table(edge_in_calico)] #these 1006 edges are due to the 34 problematic ones

D <- D[edge_in_calico == T, .(to, from)]

write.csv(D, "C:/STUDIES/RESEARCH/neural_ODE/idea_calico_data/gold_standard_edges.csv", row.names = F)
