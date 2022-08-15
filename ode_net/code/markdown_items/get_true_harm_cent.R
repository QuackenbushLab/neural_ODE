library(data.table)
library(ggplot2)
library(stringr)

print("reading in files")
effects_mat <- fread("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/effects_mat.csv")
gene_eff <- as.data.table(effects_mat)
num_genes <- dim(effects_mat)[1]
print(dim(effects_mat))
gene_eff[,reg:= .I]

print("melting and merging names")
affected_cols <- paste0("V",1:num_genes)
gene_eff <- melt(gene_eff, 
                 id.vars = "reg", measure.vars = affected_cols,
                 variable.name = "aff", value.name = "effect")
gene_eff[,aff := gsub("V","",aff)]

cell_names <- data.table(read.delim("/home/ubuntu/neural_ODE/ode_net/code/markdown_items/desmedt_gene_names_11165.csv",
                         sep = ",",
                         header = T))
print(dim(effects_mat)[1] ==nrow(cell_names))
gene_eff[,reg := cell_names[as.numeric(reg),gene]]
gene_eff[,aff := cell_names[as.numeric(aff),gene]]

print("getting true edges")
true_edges <- fread("/home/ubuntu/neural_ODE/ode_net/code/markdown_items/otter_chip_val_clean.csv")

setnames(true_edges, 
         old = c("from","to"),
         new = c("reg","aff"))
true_edges[, activation_sym := "known_edge"]
gene_eff <- merge(gene_eff, true_edges, by = c("reg","aff"), all.x = TRUE)
gene_eff[,true_effect := "no_effect"]
gene_eff[!is.na(activation_sym), true_effect := "true_effect"]

print("doing centrality stuff now")
library(igraph)
library(CINNA)

inferred_edges_main <- gene_eff[, .(reg, aff, true_effect)]
rm(gene_eff)
inferred_edges_main <- inferred_edges_main[true_effect == "true_effect"]
print(inferred_edges_main[,.N])

G_inferred <- graph_from_data_frame(inferred_edges_main, 
                                 directed = TRUE, 
                                 vertices = NULL)

h_cent_all <- harmonic_centrality(G_inferred,
                                   mode = "in")

harm_cent_to_write <- data.table(gene = names(h_cent_all), h_cent = h_cent_all)
harm_cent_to_write <- harm_cent_to_write[order(-h_cent),]
print(harm_cent_to_write)

write.csv(harm_cent_to_write, "/home/ubuntu/neural_ODE/ode_net/code/markdown_items/true_harm_cents_bc_in.csv", row.names = F)
