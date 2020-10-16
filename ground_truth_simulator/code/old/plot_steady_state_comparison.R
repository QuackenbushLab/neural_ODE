library(data.table)
library(ggplot2)

odedatamat <- fread("C:/STUDIES/RESEARCH/ODE_project/clean_data/simulated_expression_20200717.csv")
this_sample <- 1
ode_data_steady <- odedatamat[sample==1 & time == 10, ]
ode_data_steady <- melt(ode_data_steady,
                       id.vars = c("sample", "time"),
                       measure.vars = setdiff(names(ode_data_steady),
                                              c("sample","time")))
ode_data_steady <- ode_data_steady[,.(variable,value)]
setnames(ode_data_steady, 
         old = c("variable", "value"),
         new = c("node","steady_state_ode"))
ode_data_steady[, input_node := "Output Node"]
ode_data_steady[grepl("_input",node), input_node := "Input Node"]
ode_data_steady[,node := gsub("_input","",node)]

nleq_data <- fread("C:/STUDIES/RESEARCH/ODE_project/clean_data/old/simulated_expression_20200709.csv")
nleq_data_steady<- nleq_data[,.(node, sample_1_tau_1)]
setnames(nleq_data_steady,
         old = "sample_1_tau_1",
         new = "steady_state_nleq")

plot_data <- merge(ode_data_steady, nleq_data_steady, by = "node")


ggplot(data = plot_data, aes(x = steady_state_ode, 
                                  y = steady_state_nleq, 
                                  color = input_node)) + 
  geom_abline(intercept = 0, slope = 1, color = "black", lwd = 1) +
  geom_point() +
  theme_bw()
  
