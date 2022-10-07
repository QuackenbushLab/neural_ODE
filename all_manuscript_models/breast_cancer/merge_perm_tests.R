library(data.table)

range01 <- function(x){(x-min(x))/(max(x)-min(x))}

num_tops <- 20
all_genes <- c(500, 2000)
all_top_paths <- c()

print(paste0("collecting top ", num_tops," pathways from each set"))
for(this_gene in all_genes){
  this_gene_file <- paste0("C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/breast_cancer/go_mf_permtests/permtest_",
                           this_gene,".csv")
  
  D <- fread(this_gene_file)
  all_top_paths <- c(all_top_paths,
                     trimws(D[order(-phnx_z_score),][1:num_tops, pathway]))
  
}

all_top_paths <- unique(all_top_paths)
print(paste0("Collected ", length(all_top_paths), " paths!"))

#print("Check if the union exits in all gene sets")
#for(this_gene in all_genes){
#  this_gene_file <- paste0("C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/breast_cancer/harm_cents_",
#                           this_gene,".csv")
  
#  D <- fread(this_gene_file)
#  print(all(all_top_genes %in% D[, gene]))
#}


print("Now merging data for all these pathways")
all_permtest_merged <- data.table(pathway = character(),
                               z_score = numeric(),
                               num_gene = numeric())
for(this_gene in all_genes){
  this_gene_file <- paste0("C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/breast_cancer/go_mf_permtests/permtest_",
                           this_gene,".csv")
  
  D <- fread(this_gene_file)
  D <-  D[order(-phnx_z_score),]
  D_to_take <- D[trimws(pathway) %in% all_top_paths, 
                 .(pathway = trimws(pathway), z_score = phnx_z_score)]
  D_to_take[, num_gene := paste0("scale_to_", this_gene)]
  #D_to_take[, fold_permtest:= fold_permtest/this_gene] #.I
  #D_to_take[, fold_permtest:= range01(fold_permtest)] #.I
  
  all_permtest_merged <- rbind(all_permtest_merged, D_to_take)
}

all_permtest_merged[ , .N , by = num_gene]

all_permtest_wide <- dcast(all_permtest_merged, 
                          pathway ~ num_gene, value.var = "z_score")
setcolorder(all_permtest_wide, 
            c("pathway", 
              paste("scale_to", all_genes, sep = "_")))
#all_permtest_wide[,pathway := gsub(",",";", pathway)]

all_permtest_wide <- all_permtest_wide[order(-scale_to_500,
                                       -scale_to_2000,
                                       #-scale_to_4000,
                                       #-scale_to_11165
                                       )]
#all_permtest_wide <- all_permtest_wide[order(pathway)]

print(all_permtest_wide)

write.csv(all_permtest_wide,
          "C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/breast_cancer/all_permtestments_go_mf.csv",
          row.names = F,
          na = "")
