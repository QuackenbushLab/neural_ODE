library(data.table)
full_data <- readRDS("C:/STUDIES/RESEARCH/neural_ODE/mouse_single_cell_data/clean_data/ode_data_clean.rds")
all_genes <- colnames(full_data)[5:ncol(full_data)]
#expressin is measured as RPKM

most_var_genes <- full_data[,lapply(.SD, var), .SDcols = all_genes]
most_var_genes <- t(most_var_genes)
var_by_genes <- data.table(gene = row.names(most_var_genes), 
                           gene_var = most_var_genes[,1])
most_var_genes_in_dataset <- var_by_genes[order(-gene_var)][1:1000, gene]
  
cols_to_subset <- c(colnames(full_data)[2:4], most_var_genes_in_dataset)
subset_data <- full_data[,(cols_to_subset), with = F]

#remove data with missing cell number & only focus on cells with more than 1 time point
embryo_cell_combos_to_use <- subset_data[,.N, 
                                         by = .(embryo, cell)][!is.na(cell) & N>1, 
                                                               .(embryo, cell)]
samp_percentiles <- c(0.1, 0.3, 0.5, 0.7, 0.9)
nsamp <- length(samp_percentiles)
perc_column_names <- paste0("prcntl_",samp_percentiles)
netSize <- length(most_var_genes_in_dataset)

subset_data <- merge(subset_data,
                     embryo_cell_combos_to_use,
                     by = c("embryo","cell"))

subset_data[,.(num_embryo = length(unique(embryo)),num_cell = .N),by = .(stage)][order(num_cell)]

datamat <- melt(subset_data, 
          id.vars = c("embryo","cell","stage"),
          measure.vars = most_var_genes_in_dataset, 
          variable.name = "gene",
          value.name = "expression"
          )

#datamat[,distance := sqrt(sum((expression-0)^2)),
#        by = .(stage, embryo, cell)]

datamat[,distance := median(expression), #median so that we are robust to outliers
        by = .(stage, embryo, cell)]


datamat <- datamat[order(stage, distance)]

for (idx in 1:nsamp){
  this_perc <- samp_percentiles[idx]
  this_perc_colname <- perc_column_names[idx]
  print(this_perc_colname)
  datamat[,(this_perc_colname):= 
            log(expression[which(distance == quantile(unique(distance), 
                                                        this_perc, 
                                                        type = 3))]+
                  0.0001)/10,
          by = .(gene, stage)]
}

datamat <- datamat[,.SD[1], by = .(stage, gene), .SDcols = perc_column_names]

datamat <- melt(datamat, id.vars = c("gene","stage"), 
                measure.vars = perc_column_names,
                variable.name = "sample", value.name = "log_rpkm")

datamat[,var(log_rpkm),by = .(gene,stage)][,mean(V1), by = stage][order(V1)]

datamat <- dcast(datamat, sample + gene ~ stage, value.var = "log_rpkm")

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

setcolorder(datamat, neworder= c("sample","gene", stages_in_order))
write.csv(datamat, 
          "C:/STUDIES/RESEARCH/neural_ODE/mouse_single_cell_data/clean_data/mouse_SC_most_var.csv",
          row.names = F)

#datamat <- datamat[, .SD[1:(.N+1)], 
#                   by=.(embryo, cell)][is.na(gene), 
#                                       (stages_in_order) := as.list(stage_pseudo_times)]

datamat <- datamat[, .SD[1:(.N+1)],
                   by = .(sample)][is.na(gene), 
                                       (stages_in_order) := as.list(stage_pseudo_times)]

datamat[,c("gene","sample") := NULL]
top_row <- as.list(rep(NA, length(stage_pseudo_times)))
top_row[[1]] <- netSize
top_row[[2]] <- nsamp

datamat <- rbind(top_row, datamat)


write.table( datamat,
             "C:/STUDIES/RESEARCH/neural_ODE/mouse_single_cell_data/clean_data/mouse_SC_1000mostvargenes_5cellordersamples.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")
