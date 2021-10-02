library(data.table)
library(stringr)


#full_data <- read.table("C:/STUDIES/RESEARCH/ODE_project_local_old/idea_wide_format_data/idea_wide_format_data.txt",
#                        sep = "\t", header = T, quote = "")
#full_data$GENE <- as.character(full_data$GENE)
#full_data$GENE <- gsub(",","", full_data$GENE)
#full_data$GENE <- gsub("\"","", full_data$GENE)
#full_data[, (expression_cols) := lapply(.SD, as.numeric),
#          .SDcols = expression_cols]
#saveRDS(full_data, "C:/STUDIES/RESEARCH/ODE_project_local_old/idea_wide_format_data/idea_wide_format_data.rds")

full_data <- data.table(readRDS("C:/STUDIES/RESEARCH/ODE_project_local_old/idea_wide_format_data/idea_wide_format_data.rds"))
full_data[ GENE == "1-Oct", GENE  := "OCT1"]
expression_cols <- setdiff(colnames(full_data), "GENE")

expression_cols_DF <- data.table(full_name = expression_cols)
expression_cols_DF[, c("sample_id","date","reg_changed", "time"):= 
                     as.list(str_split(full_name,"_")[[1]][c(2,3,1,6)]) ,
                   by = full_name]
expression_cols_DF[ ,sample_id:= paste(sample_id, date, sep = "_"), by = full_name]
expression_cols_DF[, date := NULL]

expression_cols_DF[, all(c(0,5,10,15,20,30,45,90) %in% 
                           as.numeric(time)),
                   by = .(sample_id, reg_changed)][V1== T, ][order(reg_changed)]
#these are the 171 samples that have FULL time c(0,5,10,15,20,30,45,90)
# remove the other 46 samples for now

samples_to_keep <- expression_cols_DF[, all(c(0,5,10,15,20,30,45,90) %in% 
                                              as.numeric(time)),
                                      by = .(sample_id, reg_changed)][V1== T, sample_id]

samples_to_keep_small <- sample(samples_to_keep, 20, replace = F)
expression_cols_DF[sample_id %in% samples_to_keep &  as.numeric(time) %in% c(0,5,10,15,20,30,45,90),  
                   all(sort(as.numeric(time)) == c(0,5,10,15,20,30,45,90)),
                   by = sample_id][, all(V1)]
#done, each of these 171 samples has time exactly c(0,5,10,15,20,30,45,90)

expression_cols_DF[sample_id %in% samples_to_keep & 
                     as.numeric(time) %in% c(0,5,10,15,20,30,45,90), 
                   .N, 
                   by = .(sample_id)][order(-N)]


full_data <- melt(full_data, 
                  id.vars = "GENE",
                  measure.vars = expression_cols,
                  variable.name = "experiment",
                  value.name = "exprs_ratio")


full_data <- merge(full_data, 
                   expression_cols_DF[sample_id %in% samples_to_keep &  
                                        as.numeric(time) %in% c(0,5,10,15,20,30,45,90),],
                   by.x = "experiment", by.y = "full_name", all.y = T)

samples_to_keep_small <- full_data[reg_changed != GENE, 
          .(mean_expr = mean(abs(exprs_ratio))),
          by = .(sample_id, GENE)][,.(perc_non_zero_genes = sum(mean_expr != 0)/.N),
                                   by = sample_id][order(-perc_non_zero_genes)][1:20,sample_id]


full_data[, .N, by = sample_id][order(-N)] #makes sense 6175 gene * 8 t = 49400 per sample
full_data[, experiment := NULL]
setcolorder(full_data, c("sample_id", "reg_changed", "GENE", "time", "exprs_ratio"))

small_data <- full_data[sample_id %in% samples_to_keep_small]
small_data[, .N, by = sample_id][order(-N)] #makes sense 6175 gene * 8 t = 49400 per sample


datamat <- dcast(full_data,
                 sample_id + reg_changed + GENE ~ time,
                 value.var = "exprs_ratio")

datamat_small <- dcast(small_data,
                       sample_id + reg_changed + GENE ~ time,
                       value.var = "exprs_ratio")

all_genes <- datamat[sample_id == "SMY113_10.06.2015", GENE]
time_points <- paste("time", c(0,5,10,15,20,30,45,90), sep = "_")
setnames(datamat, 
         old = c("0.0", "05.0","10.0","15.0","20.0",
                          "30.0","45.0","90.0"),
         new = time_points
         )

setnames(datamat_small, 
         old = c("0.0", "05.0","10.0","15.0","20.0",
                 "30.0","45.0","90.0"),
         new = time_points
)


time_vals <- c(0,5,10,15,20,30,45,90)

datamat <- datamat[, .SD[1:(.N+1)], 
                   by=.(sample_id, reg_changed)][is.na(GENE), 
                                       (time_points) := as.list(time_vals)]

datamat[,c("sample_id","reg_changed","GENE") := NULL]
top_row <- as.list(rep(NA, length(time_vals)))
top_row[[1]] <- length(all_genes)
top_row[[2]] <- length(samples_to_keep)
datamat <- rbind(top_row, datamat)


datamat_small <- datamat_small[, .SD[1:(.N+1)], 
                   by=.(sample_id, reg_changed)][is.na(GENE), 
                                                 (time_points) := as.list(time_vals)]

datamat_small[,c("sample_id","reg_changed","GENE") := NULL]
datamat_small_no05 <- copy(datamat_small)
top_row <- as.list(rep(NA, length(time_vals)))
top_row[[1]] <- length(all_genes)
top_row[[2]] <- length(samples_to_keep_small)
datamat_small <- rbind(top_row, datamat_small)


datamat_small_no05[, time_0 := NULL]
datamat_small_no05[, time_5 := NULL]
top_row <- as.list(rep(NA, length(time_vals) - 2))
top_row[[1]] <- length(all_genes)
top_row[[2]] <- length(samples_to_keep_small)
datamat_small_no05<- rbind(top_row, datamat_small_no05)


write.table( datamat,
             "C:/STUDIES/RESEARCH/neural_ODE/idea_calico_data/clean_data/calico_6175genes_171samples_8T.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")

write.table( datamat_small,
             "C:/STUDIES/RESEARCH/neural_ODE/idea_calico_data/clean_data/calico_6175genes_20samples_8T.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")

write.table( datamat_small_no05,
             "C:/STUDIES/RESEARCH/neural_ODE/idea_calico_data/clean_data/calico_6175genes_20highvarsamples_6T.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")

