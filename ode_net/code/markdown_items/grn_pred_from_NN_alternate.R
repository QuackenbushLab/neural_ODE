library(data.table)
library(ggplot2)
library(stringr)

print("reading in files")
wo_prods <- fread("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/wo_prods.csv")
bo_prods <- fread("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/bo_prods.csv")
wo_sums <- fread("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/wo_sums.csv")
bo_sums <- fread("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/bo_sums.csv")
alpha_comb <- fread("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/alpha_comb.csv")

num_features <- dim(alpha_comb)[1]

print("calculating matrices")
gene_eff_sums = as.matrix(wo_sums) %*%
  as.matrix(alpha_comb[1:(num_features/2),]) 
gene_eff_prods <-  as.matrix(wo_prods) %*%
  as.matrix(alpha_comb[(num_features/2 + 1):num_features,])
rm(wo_sums)
rm(bo_sums)
rm(wo_prods)
rm(bo_prods)
gene_eff <- as.data.table(gene_eff_prods + gene_eff_sums)
num_genes <- dim(alpha_comb)[2]
rm(alpha_comb)
rm(gene_eff_sums)
rm(gene_eff_prods)
gene_eff[,reg:= .I]

print("melting and merging names & multipliers")
affected_cols <- paste0("V",1:num_genes)
gene_eff <- melt(gene_eff, 
                 id.vars = "reg", measure.vars = affected_cols,
                 variable.name = "aff", value.name = "effect")
gene_eff[,aff := gsub("V","",aff)]

cell_names <- data.table(read.delim("/home/ubuntu/neural_ODE/ode_net/code/markdown_items/gene_names.csv",
                         sep = ",",
                         header = T))
gene_eff[,reg := cell_names[as.numeric(reg),gene]]
gene_eff[,aff := cell_names[as.numeric(aff),gene]]
gene_mult <- fread("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/gene_mult.csv")
gene_mult[, multiplier := V1]
gene_mult[, gene_name := cell_names$gene]
gene_mult[, V1:= NULL]
gene_eff <- merge(gene_eff, gene_mult, by.x = "aff", by.y = "gene_name", all.x = T)
setnames(gene_eff, "multiplier", "gene_mult_of_aff")
gene_eff[, effect := (effect * gene_mult_of_aff)]
gene_eff[, gene_mult_of_aff := NULL]

print("getting pair-wise mins")
# gene_pairs_2 <- fread( "/home/ubuntu/neural_ODE/ode_net/code/markdown_items/pair_stuff.csv")
# gene_eff[, pair:= gene_pairs_2$pair]
# gene_eff[,min_effect_in_pair := min(abs(effect)),by = pair]
# gene_eff[abs(effect) == min_effect_in_pair, effect := 0]

print("calculating prop_effects")
gene_eff[, prop_effect := abs(effect)/(sum(abs(effect))), by = aff]
gene_eff[is.na(prop_effect), prop_effect :=0 ]
gene_eff[, effect := NULL]
#print(gene_eff)

print("getting true edges")
true_edges <- fread("/home/ubuntu/neural_ODE/ode_net/code/markdown_items/otter_chip_val_clean.csv")
setnames(true_edges, 
         old = c("from","to"),
         new = c("reg","aff"))
true_edges[, activation_sym := "known_edge"]
print(true_edges)
gene_eff <- merge(gene_eff, true_edges, by = c("reg","aff"), all.x = TRUE)

print("getting confusion-matrix values:")
library(PRROC)
PRROC_obj <- roc.curve(scores.class0 = gene_eff$prop_effect,
                      weights.class0 = !is.na(gene_eff$activation_sym),
                       curve=TRUE)

best_index <- which.max(1-PRROC_obj$curve[,1]+PRROC_obj$curve[,2])
prop_cut_off <- PRROC_obj$curve[best_index,3]
gene_eff[, .(avg_pred_effect = mean(prop_effect), .N),
         by = .(activation_sym)]

gene_eff[,pred_effect := "no_effect"]
gene_eff[prop_effect > prop_cut_off, 
         pred_effect := "pred_effect"]
gene_eff[,true_effect := "no_effect"]
gene_eff[!is.na(activation_sym), true_effect := "true_effect"]

gene_eff[,prop.table(table(true_effect, pred_effect), margin = 1)]
print(PRROC_obj$auc)

# library(igraph)
# library(CINNA)

# inferred_edges <- gene_eff[, .(reg, aff, prop_effect)]
# prop_cut_off <- as.numeric(inferred_edges[, 
#                                           quantile(prop_effect, 
#                                                    0.995, 
#                                                    na.rm = T)])
# inferred_edges_main <- inferred_edges[prop_effect > prop_cut_off]

# G_inferred <- graph_from_data_frame(inferred_edges_main, 
#                                 directed = TRUE, 
#                                 vertices = NULL)

# h_cent_all <- harmonic_centrality(G_inferred,
#                                   mode = "out",
#                                   weights = 1/inferred_edges_main$prop_effect)

# harm_cent_plot <- data.table(gene = names(h_cent_all), h_cent = h_cent_all)
# plot_subset <- harm_cent_plot[order(-h_cent)][1:15]
# ggplot(plot_subset, 
#        aes(x = factor(gene,
#                       levels = plot_subset$gene), 
#            y = h_cent)) + 
#   geom_col(fill = "dodgerblue") +
#   theme_bw() + 
#   theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
#   xlab("most influential genes")
