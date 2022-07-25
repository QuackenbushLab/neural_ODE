library(data.table)
library(stringr)

### marker genes
marker_genes <- fread("C:/STUDIES/RESEARCH/neural_ODE/breast_cancer_data/clean_data/desmedt_gene_names_4000.csv")
marker_genes <- marker_genes$gene #2000 marker genes

### get full_data
full_data <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/Data_smooth_allgenes.csv")
full_genes <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/GeneNames.csv", header = F)
names(full_genes) <- c("gene")
full_data <- cbind(full_genes, full_data)
full_data <- full_data[!is.na(gene) & gene != "", ]
full_data <- full_data[,.SD[1], by = gene]
full_data <- full_data[gene %in% marker_genes]
full_data <- full_data[order(gene)]

### get ER pos data
erpos_data <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/Breast cancer-ER subtypes/er=1/er1_Ordered_Data.csv")
erpos_data <- erpos_data[-.N] #remove grade row
setnames(erpos_data, "V1", "gene")
erpos_data <- erpos_data[gene %in% marker_genes]
erpos_data <- erpos_data[order(gene)]

### find best matches
all(full_data$gene == erpos_data$gene)
sample_ids <- tail(names(erpos_data), -1)
full_data_exprs_cols <- paste0("V", 1:186)

for (this_sample in sample_ids){
  this_sample_erpos_exprs <- erpos_data[[this_sample]]
  best_matches <- names(tail(sort((full_data[,sapply(.SD, function(vec){cor(vec, this_sample_erpos_exprs)}),
                                           .SDcols = full_data_exprs_cols])), 3))
  print(paste(this_sample,"=",paste0(best_matches, collapse = ", ")))
}

