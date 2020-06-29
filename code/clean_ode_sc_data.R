library(data.table)
raw_data_dir <- "C:/STUDIES/RESEARCH/ODE_project/raw_data/GSE45719_RAW"
cell_files <- list.files(path = raw_data_dir,full.names = FALSE, 
                         recursive = FALSE, include.dirs =FALSE)

## extract individual files and consolidate into one object (final_DT)
counter <- 0
for(this_file in cell_files){
  counter <- print(counter + 1)
  
  thisDT <- data.table(read.table(gzfile(paste(raw_data_dir, this_file,sep = "/")),
                 sep = "\t",header = FALSE))
  
  thisDT <- thisDT[,c("V1","V3")]
  names(thisDT) <- c("gene","rpkm")
  thisDT <- thisDT[,.(rpkm = mean(rpkm, na.rm = T)), by = gene]
  thisDT[,sample := this_file]
  X  = data.table::dcast(thisDT, sample ~ gene, value.var = "rpkm")
  
  if (counter ==1){
    final_DT <- X
  }
  else{
    final_DT <- rbind(final_DT, X)
  }
}


## some final post-processing
final_DT[,`:=`(stage = strsplit(sample,"_")[[1]][2],
               embryo = strsplit(strsplit(sample,"_")[[1]][3],"-")[[1]][1],
               cell = strsplit(strsplit(sample,"_")[[1]][3],"-")[[1]][2]
               ), by = sample]

col_idx <- which(names(final_DT) %in% c("sample","stage","embryo","cell"))
new_order = names(final_DT)[c(col_idx, (1:ncol(final_DT))[-col_idx])]
setcolorder(final_DT, neworder  = new_order)
final_DT[,sample:= .I]

## take a look 
final_DT[1:5, 1:7]
dim(final_DT)

## write files
saveRDS(final_DT, "C:/STUDIES/RESEARCH/ODE_project/clean_data/ode_data_clean.rds")
write.csv(final_DT, "C:/STUDIES/RESEARCH/ODE_project/clean_data/ode_data_clean.csv",
          row.names = FALSE)

