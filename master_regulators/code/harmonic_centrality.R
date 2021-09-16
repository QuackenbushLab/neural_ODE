library(data.table)
library(ggplot2)
library(igraph)
library(CINNA)
library(influential)


#chief_directory <- "/home/ubuntu/neural_ODE/master_regulators/"
chief_directory <- "C:/STUDIES/RESEARCH/neural_ODE/master_regulators"
write_directory <- paste(chief_directory,"score_outputs/scores_to_save_avoid0.csv", sep = "/")

true_edges <- fread(paste(chief_directory,"model_to_test/edge_properties.csv", sep = "/"))
setnames(true_edges, 
         old = c("from","to"),
         new = c("reg","aff"))

G_true <- graph_from_data_frame(true_edges, 
                                directed = TRUE, 
                                vertices = NULL)

h_cent_out <- harmonic_centrality(G_true,
                              mode = "out",
                              weights = NULL)

h_cent_in_out <- harmonic_centrality(G_true,
                                  mode = "all",
                                  weights = NULL)

ivi_cent <- ivi(graph = G_true, 
                       weights = NULL, 
                        directed = TRUE, 
                       mode = "out", d = 3)


all(names(h_cent_in_out) == names(h_cent_out))
all(names(h_cent_in_out) == names(ivi_cent))



D_cent <- data.table(gene = names(h_cent_in_out), 
                     out_cent = h_cent_out,
                     out_in_cent = h_cent_in_out,
                     ivi_cent = ivi_cent)

write.csv(D_cent, 
          "C:/STUDIES/RESEARCH/neural_ODE/master_regulators/model_to_test/gene_centralities.csv",
          row.names = F)
