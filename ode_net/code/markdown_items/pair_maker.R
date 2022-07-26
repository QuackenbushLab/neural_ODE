library(data.table)

label_maker <- function(lab_1, lab_2){
  this_vec <- sort(c(lab_1, lab_2))
  paste(this_vec, collapse = "_")
}

pair_maker <- function(num_gene){
  D <- data.table(expand.grid(1:num_gene, 1:num_gene))
  D <- D[order(Var1, Var2)]
  D[, pair:= mapply(label_maker, Var1, Var2)]
  return(D[, .(pair)])
}

system.time(pair_stuff <- pair_maker(2000))
print(pair_stuff)
#write.csv(pair_stuff,"/home/ubuntu/neural_ODE/ode_net/code/markdown_items/pair_stuff_11165.csv", row.name = F)