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

cell_names <- data.table(read.delim("/home/ubuntu/neural_ODE/ode_net/code/markdown_items/pramila_gene_names.csv",
                         sep = ",",
                         header = T))
cell_names[,x:= gsub("_input","", x)]
setnames(cell_names, old = "x", new = "gene")


print(dim(effects_mat)[1] ==nrow(cell_names))
gene_eff[,reg := cell_names[as.numeric(reg),gene]]
gene_eff[,aff := cell_names[as.numeric(aff),gene]]

print("calculating prop_effects")
gene_eff[, prop_effect := abs(effect)/(sum(abs(effect))), by = aff]
gene_eff[is.na(prop_effect), prop_effect :=0 ]
gene_eff[, effect := NULL]

print("getting true edges")
true_edges <- fread("/home/ubuntu/neural_ODE/ode_net/code/markdown_items/harbison_chip_orf.csv")

true_edges <- true_edges[p_val < 0.001,]
true_edges[, num_edges_for_this_TF := .N, by = from]

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
                       curve= F) #curve = TRUE

gene_eff[, .(avg_pred_effect = mean(prop_effect), .N),
           by = .(activation_sym)]

print(paste("AUC =", PRROC_obj$auc))

quit(status=1)

this_curve <- PRROC_obj$curve

idx <- c(1:100, 
         sort(sample(100:(nrow(this_curve)-100), 100, replace = F)),
         (nrow(this_curve)-100): nrow(this_curve)
         )

curve_to_write <- this_curve[idx,]
curve_to_write[,1] <- curve_to_write[,1]
colnames(curve_to_write) <- c("fpr", "tpr", "cutoff")
write.csv(curve_to_write, 
          "/home/ubuntu/neural_ODE/ode_net/code/markdown_items/auc_curve.csv",
          row.names = F)


#

