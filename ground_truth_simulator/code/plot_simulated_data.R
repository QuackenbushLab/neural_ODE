library(data.table)
library(ggplot2)

fulldatamat <- fread("C:/STUDIES/RESEARCH/ODE_project/clean_data/simulated_expression_20200717.csv")
this_sample <- 1
plot_data <- fulldatamat[sample == this_sample,
                         c(2,sample(3:length(fulldatamat),20)),
                         with = F]
plot_data_melt <- melt(plot_data, id.vars = c("time"),
     measure.vars = setdiff(names(plot_data),"time"))
setnames(plot_data_melt, 
         old = c("variable", "value"),
         new = c("gene","expression"))

plot_data_melt[,input_gene := "Output gene"]
plot_data_melt[ grepl("_input",gene), 
                input_gene := "Input gene"]

ggplot(data = plot_data_melt, aes(x = time, 
                                  y = expression, 
                                  color = gene)) + 
  geom_point()+ 
  geom_line(aes(group = gene))+
  theme_bw() + 
  facet_grid(.~input_gene)

