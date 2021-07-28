library(data.table)
full_data <- readRDS("C:/STUDIES/RESEARCH/neural_ODE/mouse_single_cell_data/clean_data/ode_data_clean.rds")
all_genes <- colnames(full_data)[5:ncol(full_data)]

cell_cycle_genes <- fread("C:/STUDIES/RESEARCH/neural_ODE/mouse_single_cell_data/code/GO_mouse_cell_cycle_genes.csv")
cell_cycle_genes <- cell_cycle_genes$symbol

cell_cycle_genes_in_dataset <- cell_cycle_genes[which(cell_cycle_genes %in% all_genes)]
cols_to_subset <- c(colnames(full_data)[2:4], cell_cycle_genes_in_dataset)
subset_data <- full_data[,(cols_to_subset), with = F]

#remove data with missing cell number & only focus on cells with more than 1 time point
embryo_cell_combos_to_use <- subset_data[,.N, 
                                         by = .(embryo, cell)][!is.na(cell) & N>1, 
                                                               .(embryo, cell)]
nsamp <- nrow(embryo_cell_combos_to_use)
netSize <- length(cell_cycle_genes_in_dataset)

subset_data <- merge(subset_data,
                     embryo_cell_combos_to_use,
                     by = c("embryo","cell"))

datamat <- melt(subset_data, 
          id.vars = c("embryo","cell","stage"),
          measure.vars = cell_cycle_genes_in_dataset, 
          variable.name = "gene",
          value.name = "expression"
          )

datamat <- dcast(datamat, embryo + cell + gene ~ stage, value.var = "expression")

stages_in_order <- c("early2cell", "mid2cell", "late2cell",
                      "4cell", "8cell", "16cell",
                     "earlyblast","midblast", "lateblast")

stage_pseudo_times <- c(1,1.5, 2,
                        2.2, 2.5, 3,
                        3.2, 4, 4.5) 
#NEED TO FIGURE OUT PSEUDOtimes
#using for now: https://www.emouseatlas.org/emap/ema/theiler_stages/StageDefinition/stagedefinition.html#dpc

setcolorder(datamat, neworder= c("embryo","cell","gene", stages_in_order))
write.csv(datamat, 
          "C:/STUDIES/RESEARCH/neural_ODE/mouse_single_cell_data/clean_data/mouse_SC_cell_cycle.csv",
          row.names = F)

datamat <- datamat[, .SD[1:(.N+1)], 
                   by=.(embryo, cell)][is.na(gene), 
                                       (stages_in_order) := as.list(stage_pseudo_times)]
datamat[,c("embryo","cell","gene") := NULL]
top_row <- as.list(rep(NA, length(stage_pseudo_times)))
top_row[[1]] <- netSize
top_row[[2]] <- nsamp

datamat <- rbind(top_row, datamat)


write.table( datamat,
             "C:/STUDIES/RESEARCH/neural_ODE/mouse_single_cell_data/clean_data/mouse_SC_1010genes_72samples.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")
