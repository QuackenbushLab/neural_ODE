library(data.table)

range01 <- function(x){(x-min(x))/(max(x)-min(x))}

num_tops <- 10
all_genes <- c(500, 2000, 4000, 11165)
all_top_paths <- c()

print(paste0("collecting top ", num_tops," pathways from each set"))
for(this_gene in all_genes){
  this_gene_file <- paste0("C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/breast_cancer/go_molec_enrichment_200_most_central/enrichment_",
                           this_gene,".csv")
  
  D <- fread(this_gene_file)
  names(D) <- gsub(" ","_", names(D))
  n_g <- length(unique(trimws(strsplit(paste(D$Genes, collapse = "")," ")[[1]])))
  print(paste("For dataset=", this_gene, ", I see", n_g, "genes that participate in pathways"))
  all_top_paths <- c(all_top_paths,
                     trimws(D[order(-Fold_Enrichment),][1:num_tops, Pathway]))
  
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
all_enrich_merged <- data.table(pathway = character(),
                               fold_enrich = numeric(),
                               num_gene = numeric())
for(this_gene in all_genes){
  this_gene_file <- paste0("C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/breast_cancer/go_molec_enrichment_200_most_central/enrichment_",
                           this_gene,".csv")
  
  D <- fread(this_gene_file)
  names(D) <- gsub(" ","_", names(D))
  D <-  D[order(-Fold_Enrichment),]
  D_to_take <- D[trimws(Pathway) %in% all_top_paths, 
                 .(pathway = trimws(Pathway), fold_enrich = Fold_Enrichment)]
  D_to_take[, num_gene := paste0("scale_to_", this_gene)]
  #D_to_take[, fold_enrich:= fold_enrich/this_gene] #.I
  #D_to_take[, fold_enrich:= range01(fold_enrich)] #.I
  
  all_enrich_merged <- rbind(all_enrich_merged, D_to_take)
}

all_enrich_merged[ , .N , by = num_gene]

all_enrich_wide <- dcast(all_enrich_merged, 
                          pathway ~ num_gene, value.var = "fold_enrich")
setcolorder(all_enrich_wide, 
            c("pathway", 
              paste("scale_to", all_genes, sep = "_")))
all_enrich_wide[,pathway := gsub(",",";", pathway)]

all_enrich_wide <- all_enrich_wide[order(-scale_to_500,
                                       -scale_to_2000,
                                       -scale_to_4000,
                                       -scale_to_11165
                                       )]
#all_enrich_wide <- all_enrich_wide[order(pathway)]

print(all_enrich_wide)

write.csv(all_enrich_wide,
          "C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/breast_cancer/all_enrichments_go_molec_200central_wide.csv",
          row.names = F,
          na = "")
