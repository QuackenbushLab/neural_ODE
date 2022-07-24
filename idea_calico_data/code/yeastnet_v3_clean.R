library(data.table)

D = read.table("C:/STUDIES/RESEARCH/ODE_project_local_old/costanzo_yeastnet/CX.Cellcycle.YeastNet.3462gene.10702link.txt")
D = data.table(D)

names(D) <- c("from", "to", "int_score")

translate_dict <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/costanzo_yeastnet/orftogene_yeastnetv3.csv")

D <- merge(D, translate_dict, by.x = "from", by.y = "orf")
D[, from := std_name]
D[, std_name := NULL]


D <- merge(D, translate_dict, by.x = "to", by.y = "orf")
D[, to := std_name]
D[, std_name := NULL]


all_genes <- fread("C:/STUDIES/RESEARCH/neural_ODE/idea_calico_data/gene_names_6175.csv")

prob_genes <- unique(c(unique(D$from[!D$from %in% all_genes$x]), unique(D$to[!D$to %in% all_genes$x])))


D[, edge_in_calico := to %in% all_genes$x & from %in% all_genes$x, 
  by = .(to, from)]

D[, table(edge_in_calico)] #these 1006 edges are due to the 34 problematic ones

D <- D[edge_in_calico == T, .(to, from, int_score)]

write.csv(D, "C:/STUDIES/RESEARCH/neural_ODE/idea_calico_data/yeast_net_cellcycle_edges.csv", row.names = F)
