library(data.table)
library(zoo)
library(ggplot2)

my_LOCF <- function(vec){
  new_vec <- na.locf(vec, na.rm = FALSE) #LOCF
  if(is.na(new_vec[1])){
    new_vec[1] <- vec[2] #NOCB
  }
  return(new_vec)  
}


full_data <- fread("C:/STUDIES/RESEARCH/neural_ODE/rIFNbeta_data/raw_rifnb_data.csv")
all_genes <- colnames(full_data)[6:ncol(full_data)]
all_genes <- toupper(gsub("\\s","_",gsub("-|[()]","",all_genes)))

colnames(full_data)[6:ncol(full_data)] <- all_genes
#expressin is measured as RPKM

subset_genes <- fread("C:/STUDIES/RESEARCH/neural_ODE/rIFNbeta_data/gene_names.csv")
subset_genes <- subset_genes$Symbol

length(subset_genes)


setdiff( subset_genes, all_genes)
setdiff(all_genes, subset_genes)

full_data[,mean(edss), by = response_status]
full_data[,.N == length(unique(sample_id))]

patient_conditions <- full_data[,.(pid, time, response_status, edss)]


full_data[, time := as.numeric(gsub("t","",time))]


full_data_melt <- melt(full_data, 
                       id.vars = c("pid","time","response_status"),
                       measure.vars = subset_genes, #subset to 70 in paper
                       variable.name = "gene",
                       value.name = "expression",
                       na.rm = FALSE
                      )
full_data_melt[is.na(expression), .N] #these need interpolation

#do interpolation for these 98 randomly missing ones
full_data_melt[, 
               expression_LOCF := my_LOCF(expression),
               by = .(pid, gene)] 

full_data_melt[, log_med_exp := log(expression_LOCF) - 
                 median(log(expression_LOCF), na.rm = T)]
full_data_melt[log_med_exp <= -8, log_med_exp := -8] #truncate small values
full_data_melt[is.na(log_med_exp)]
hist(full_data_melt$log_med_exp)

time_fake_gene <- full_data[,.(pid, response_status, time)]
time_fake_gene[, gene:= "ZZZ_time_col"]
time_fake_gene[, log_med_exp:= time]

full_data_melt <- merge(full_data_melt, time_fake_gene, 
                        by = c("pid","response_status","time","gene","log_med_exp"),
                        all = T)

full_data_melt[, avail_time := paste("avail",1:.N, sep = "_"), 
               by = .(pid, response_status, gene)]

ggplot(full_data_melt[gene %in% c("BAX","CASPASE_1", "TYK2","STAT6")], 
       aes(x = time, y = log_med_exp)) + 
  geom_line(aes(group = pid, col = pid)) + 
  facet_grid(gene~ response_status)

datamat <- dcast(full_data_melt,
                               pid + response_status + gene ~ avail_time,
                               valu2e.var = "log_med_exp")

#datamat <- datamat[, .SD[1:(.N+1)], 
#                   by=.(embryo, cell)][is.na(gene), 
#                                       (stages_in_order) := as.list(stage_pseudo_times)]

num_time <- 7
num_genes <- length(unique(datamat$gene)) - 1
num_traj <- length(unique(datamat$pid))
num_traj_good <- datamat[response_status == "good", length(unique(pid))]
num_traj_bad <- datamat[response_status == "bad", length(unique(pid))]


datamat_good <- datamat[response_status == "good"]
datamat_bad <- datamat[response_status == "bad"]

datamat[,c("pid","response_status","gene") := NULL]
top_row <- as.list(rep(NA, num_time))
top_row[[1]] <- num_genes
top_row[[2]] <- num_traj
datamat <- rbind(top_row, datamat)

datamat_good[,c("pid","response_status","gene") := NULL]
top_row <- as.list(rep(NA, num_time))
top_row[[1]] <- num_genes
top_row[[2]] <- num_traj_good
datamat_good <- rbind(top_row, datamat_good)

datamat_bad[,c("pid","response_status","gene") := NULL]
top_row <- as.list(rep(NA, num_time))
top_row[[1]] <- num_genes
top_row[[2]] <- num_traj_bad
datamat_bad <- rbind(top_row, datamat_bad)



write.csv(patient_conditions,
          "C:/STUDIES/RESEARCH/neural_ODE/rIFNbeta_data/clean_data/patient_conditions.csv",
          row.names = F)

write.table( datamat,
             "C:/STUDIES/RESEARCH/neural_ODE/rIFNbeta_data/clean_data/rIFNbeta_70genes_53samples.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")

write.table( datamat_good,
             "C:/STUDIES/RESEARCH/neural_ODE/rIFNbeta_data/clean_data/rIFNbeta_70genes_33samples_good.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")

write.table( datamat_bad,
             "C:/STUDIES/RESEARCH/neural_ODE/rIFNbeta_data/clean_data/rIFNbeta_70genes_20samples_bad.csv", 
             sep=",",
             row.names = FALSE,
             col.names = FALSE,
             na = "")

