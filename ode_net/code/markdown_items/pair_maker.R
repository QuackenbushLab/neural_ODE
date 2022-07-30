library(data.table)
library(progress)
num_genes <- 11165
pb <- progress_bar$new(format = "(:spin) [:bar] :percent [Elapsed time: :elapsedfull || Estimated time remaining: :eta]",
                       total = num_genes,
                       complete = "=",   # Completion bar character
                       incomplete = "-", # Incomplete bar character
                       current = ">",    # Current bar character
                       clear = FALSE,    # If TRUE, clears the bar when finish
                       width = 100) 

label_maker <- function(lab_1, lab_2){
  if(lab_2 == num_genes){
    pb$tick()
    #print(lab_1/num_genes*100)
  }
  this_vec <- sort(c(lab_1, lab_2))
  paste(this_vec, collapse = "_")
}

label_maker_vec <- function(lab_1_vec, lab_2_vec){
  return(mapply(label_maker, lab_1_vec, lab_2_vec))
}

pair_maker <- function(start_gene, num_gene){
  D <- data.table(expand.grid(start_gene :(start_gene + 999), 1:num_gene))
  D <- D[order(Var1, Var2)]
  D[, pair:= label_maker_vec(Var1, Var2)]
  return(D[, .(pair)])
}

thousands <- (0:10)*1000 +1
for (i in thousands){
  system.time(pair_stuff <- pair_maker(i, num_genes))
  print(pair_stuff)
  file_name <- paste0(pair_stuff,"_",i,".csv")
  write.csv(pair_stuff,
            paste0("/home/ubuntu/neural_ODE/ode_net/code/markdown_items/",file_name),
            row.name = F)
}
