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
#cell_names[,x:= gsub("_input","", x)]
#setnames(cell_names, old = "x", new = "gene")


print(dim(effects_mat)[1] ==nrow(cell_names))
gene_eff[,reg := cell_names[as.numeric(reg),gene]]
gene_eff[,aff := cell_names[as.numeric(aff),gene]]

print("calculating prop_effects")
gene_eff[, prop_effect := abs(effect)/(sum(abs(effect))), by = aff]
gene_eff[is.na(prop_effect), prop_effect :=0 ]
gene_eff[, effect := NULL]

print("getting true edges")
true_edges <- fread("/home/ubuntu/neural_ODE/ode_net/code/markdown_items/otter_chip_val_clean.csv")

#true_edges <- true_edges[p_val < 0.001,]
#true_edges[, num_edges_for_this_TF := .N, by = from]
#true_edges <- true_edges[num_edges_for_this_TF > 4000,]
#print(true_edges[order(-num_edges_for_this_TF)])

setnames(true_edges, 
         old = c("from","to"),
         new = c("reg","aff"))
true_edges[, activation_sym := "known_edge"]
gene_eff <- merge(gene_eff, true_edges, by = c("reg","aff"), all.x = TRUE)

#gene_eff[,reg:= NULL]
#gene_eff[,aff:= NULL]


print("getting confusion-matrix values")
library(PRROC)
PRROC_obj <- roc.curve(scores.class0 = gene_eff$prop_effect,
                      weights.class0 = !is.na(gene_eff$activation_sym),
                       curve= TRUE) #curve = TRUE

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
# png(file = "AUC_plot.png")
# plot(PRROC_obj, main = "", legend = F, col = "red")
# abline(0,1)
# dev.off()

# print("doing centrality stuff now")
# library(igraph)
# library(CINNA)

#  inferred_edges <- gene_eff[, .(reg, aff, prop_effect)]
#  rm(gene_eff)
#  prop_cut_off <- as.numeric(inferred_edges[, 
#                                            quantile(prop_effect, 
#                                                     0.995, 
#                                                     na.rm = T)])
#  inferred_edges_main <- inferred_edges[prop_effect > prop_cut_off]

#  G_inferred <- graph_from_data_frame(inferred_edges_main, 
#                                  directed = TRUE, 
#                                  vertices = NULL)

#  h_cent_all <- harmonic_centrality(G_inferred,
#                                    mode = "out",
#                                    weights = 1/inferred_edges_main$prop_effect)

#  harm_cent_plot <- data.table(gene = names(h_cent_all), h_cent = h_cent_all)

#  harm_cent_to_write <- harm_cent_plot[order(-h_cent),][1:100]
#  write.csv(harm_cent_to_write, "/home/ubuntu/neural_ODE/ode_net/code/markdown_items/harm_cents.csv", row.names = F)


#  plot_subset <- harm_cent_plot[order(-h_cent)][1:15]

#  png(file = "cent_plot.png")
#  ggplot(plot_subset, 
#         aes(x = factor(gene,
#                        levels = plot_subset$gene), 
#             y = h_cent)) + 
#    geom_col(fill = "dodgerblue") +
#    theme_bw() + 
#    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
#    xlab("most influential genes")
# dev.off()
