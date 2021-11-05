library(data.table)
sga <- read.table("C:/STUDIES/RESEARCH/ODE_project_local_old/costanzo/SGA_ExE.txt")
sga <- data.table(sga)
sga <- sga[,.(V2, V4, V6, V7)]
names(sga) <- c("query_name","array_name","int_score","p_val")
sga[, query_name:= toupper(gsub("-.*","",query_name))]
sga[, array_name:= toupper(gsub("-.*","",array_name))]
sga[, pair :=  paste(sort(c(query_name, array_name)), collapse = ", "),
         by = .(query_name, array_name)]
sig_pairs <- unique(sga[p_val < 0.05, .(pair)])
write.csv(sig_pairs,
          "C:/STUDIES/RESEARCH/neural_ODE/idea_calico_data/costanzo_edges.csv" ,
          row.names = F)
