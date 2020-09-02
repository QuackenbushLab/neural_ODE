library(data.table)
library(ggplot2)

edge_list <- fread("C:/STUDIES/RESEARCH/ODE_project/clean_data/edge_properties.csv")
chosen_gene <- edge_list[,.N, by = to][order(-N),][1,to]
incoming_edges <- edge_list[to == chosen_gene, .(from, activation)]
activators <- incoming_edges[activation == T, from]
repressors <- incoming_edges[activation == F, from]

fulldatamat <- fread("C:/STUDIES/RESEARCH/ODE_project/clean_data/simulated_expression_20200717.csv")
input_genes <- gsub("_input","",grep("_input", names(fulldatamat), value = T))
activators[activators %in% input_genes] <- paste(activators[activators %in% input_genes],
                                                 "input", sep = "_")

repressors[repressors %in% input_genes] <- paste(repressors[repressors %in% input_genes],
                                                 "input", sep = "_")

this_sample <- 1

plot_data <- fulldatamat[sample == this_sample,
                         c("time",chosen_gene,activators, repressors), with = F]

plot_data_melt <- melt(plot_data, id.vars = c("time"),
     measure.vars = setdiff(names(plot_data),"time"))
setnames(plot_data_melt, 
         old = c("variable", "value"),
         new = c("gene","expression"))

plot_data_melt[,type := paste("Output gene",chosen_gene, sep = ": ")]
plot_data_melt[ gene %in% activators, 
                type := "Activators"]
plot_data_melt[ gene %in% repressors, 
                type := "Repressors"]


ggplot(data = plot_data_melt, aes(x = time, 
                                  y = expression, 
                                  color = gene)) + 
  geom_point()+
  geom_line(aes(group = gene))+
  theme_bw() + 
  facet_grid(.~type)

