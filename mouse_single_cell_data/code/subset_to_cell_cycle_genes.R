library(data.table)
full_data <- readRDS("C:/STUDIES/RESEARCH/neural_ODE/mouse_single_cell_data/clean_data/ode_data_clean.rds")
all_genes <- colnames(full_data)[5:ncol(full_data)]
#expressin is measured as RPKM

cell_cycle_genes <- fread("C:/STUDIES/RESEARCH/neural_ODE/mouse_single_cell_data/code/GO_mouse_cell_cycle_genes.csv")
cell_cycle_genes <- cell_cycle_genes$symbol

cell_cycle_genes_in_dataset <- cell_cycle_genes[which(cell_cycle_genes %in% all_genes)]
cols_to_subset <- c(colnames(full_data)[2:4], cell_cycle_genes_in_dataset)
subset_data <- full_data[,(cols_to_subset), with = F]

#remove data with missing cell number & only focus on cells with more than 1 time point
embryo_cell_combos_to_use <- subset_data[,.N, 
                                         by = .(embryo, cell)][!is.na(cell) & N>1, 
                                                               .(embryo, cell)]
nsamp <- 1 
netSize <- length(cell_cycle_genes_in_dataset)

subset_data <- merge(subset_data,
                     embryo_cell_combos_to_use,
                     by = c("embryo","cell"))

subset_data[,.(num_embryo = length(unique(embryo)),num_cell = .N),by = .(stage)][order(num_cell)]

datamat <- melt(subset_data, 
          id.vars = c("embryo","cell","stage"),
          measure.vars = cell_cycle_genes_in_dataset, 
          variable.name = "gene",
          value.name = "expression"
          )

#datamat <- datamat[,.(mean_expression = mean(expression)), by = .(embryo, stage, gene)]
datamat <- datamat[,.(log_median_rpkm = log(median(expression)+0.0001)/10), #scale for -1 to 1
                   by = .(stage, gene)][order(stage),]


datamat <- dcast(datamat, gene ~ stage, value.var = "log_median_rpkm")

stages_in_order <- c("early2cell", "mid2cell", "late2cell",
                      "4cell", "8cell", "16cell",
                     "earlyblast","midblast", "lateblast")

stage_pseudo_times <- c(31.5,39.5, 47,
                        55, 69, 77,
                        87, 93, 101)  #hours

stage_pseudo_times <- (stage_pseudo_times - min(stage_pseudo_times))
#stage_pseudo_times <- stage_pseudo_times/max(stage_pseudo_times)*10
#start at 0 end at 10

#Embryos at different stages of preimplantation development were collected 
#at defined time periods after hCG administration: 20-24 h after hCG (zygotes), 
#31-32 h (early 2-cell), 39-40 h (mid 2-cell), 46-48 h (late 2-cell), 
#54-56 h (4-cell), 68-70 h (8-cell), 76-78 h (16-cell), 86-88 h (early blastocysts),
#92-94 h (mid blastocysts), and 100-102 h (late blastocysts)

setcolorder(datamat, neworder= c("gene", stages_in_order))
write.csv(datamat, 
          "C:/STUDIES/RESEARCH/neural_ODE/mouse_single_cell_data/clean_data/mouse_SC_cell_cycle.csv",
          row.names = F)

#datamat <- datamat[, .SD[1:(.N+1)], 
#                   by=.(embryo, cell)][is.na(gene), 
#                                       (stages_in_order) := as.list(stage_pseudo_times)]

datamat <- datamat[, .SD[1:(.N+1)]][is.na(gene), 
                                       (stages_in_order) := as.list(stage_pseudo_times)]

datamat[,c("gene") := NULL]
top_row <- as.list(rep(NA, length(stage_pseudo_times)))
top_row[[1]] <- netSize
top_row[[2]] <- nsamp

datamat <- rbind(top_row, datamat)


write.table( datamat,
             "C:/STUDIES/RESEARCH/neural_ODE/mouse_single_cell_data/clean_data/mouse_SC_1010genes_1mediansample.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")
