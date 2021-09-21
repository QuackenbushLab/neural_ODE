library(data.table)
library(ggplot2)
library(igraph)
library(CINNA)
library(influential)


#chief_directory <- "/home/ubuntu/neural_ODE/master_regulators/"
chief_directory <- "C:/STUDIES/RESEARCH/neural_ODE/master_regulators"

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



inferred_edges <- fread(paste(chief_directory,"model_to_test/model_param_based_effs.csv", sep = "/"))
inferred_edges_main <- inferred_edges[abs_effect > 2*cutoff]
inferred_edges_main[,.N]

G_inferred <- graph_from_data_frame(inferred_edges_main, 
                                directed = TRUE, 
                                vertices = NULL)

h_cent_inferred <- harmonic_centrality(G_inferred,
                                  mode = "out",
                                  weights = NULL)

D_inferred <- data.table(gene = names(h_cent_inferred), 
                         infer_cent = h_cent_inferred)

D_cent <- data.table(gene = names(h_cent_in_out), 
                     out_cent = h_cent_out,
                     out_in_cent = h_cent_in_out,
                     ivi_cent = ivi_cent)

D_cent <- merge(D_cent, D_inferred, by = "gene", all.x = T)
D_cent[is.na(infer_cent), infer_cent := 0 ]


#D_cent[,plot(out_cent, infer_cent)]
#D_cent[,cor(out_cent, infer_cent)]

write.csv(D_cent, 
          "C:/STUDIES/RESEARCH/neural_ODE/master_regulators/model_to_test/gene_centralities.csv",
          row.names = F)
