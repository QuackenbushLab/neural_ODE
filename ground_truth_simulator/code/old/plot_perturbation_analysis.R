library(data.table)
library(ggplot2)

datamat <- fread("C:/STUDIES/RESEARCH/ODE_project/clean_data/perturbation_analysis_20200720.csv")
datamat <- datamat[time %in% c(0,1,2,10),]
datamat <- melt(datamat,
     id.vars = c("sample", "time"),
     measure.vars = setdiff(names(datamat),
                            c("sample","time")))
samp_1_datamat <- datamat[sample ==1,]
samp_1_datamat[,sample:= NULL]
setnames(samp_1_datamat, old = "value", new = "samp_1_value")
plot_data <- merge(datamat, samp_1_datamat, by = c("time","variable"))
setnames(plot_data, 
         old = c("variable","value","samp_1_value"),
         new = c("node","expression","sample_1_expression"))

plot_data[, input_node := "Output Node"]
plot_data[grepl("_input",node), input_node := "Input Node"]
plot_data[,node := gsub("_input","",node)]
plot_data[, time:= paste("Time", time, sep = " = ")]
plot_data[, sample:= paste("Sample", sample, sep = ":")]
plot_data$time_f = factor(plot_data$time, 
                          levels=c("Time = 0", 
                                   "Time = 1",
                                   "Time = 2",
                                   "Time = 10"))

ggplot(data = plot_data, aes(x = sample_1_expression, 
                                  y = expression, 
                                  color = input_node)) + 
  geom_abline(intercept = 0, slope = 1, color = "black", lwd = 1) +
  geom_point() +
  theme_bw() + 
  facet_grid(sample~time_f)
  
