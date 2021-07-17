library(data.table)
E = seq(0.4, 0.6, by = 0.01)
n = seq(1.39, 1.8, by = 0.01)
D = data.table(expand.grid(E, n ))
names(D) <- c("E", "n")
D[,K_to_n := 1/(E^(-n)-2)]
hist(D$K_to_n)
D[,minus_log_K_to_n := -log(1/(E^(-n)-2))]
hist(D$minus_log_K_to_n)
D[,beta := (E^n-1)/(2*E^n-1)]
hist(D$beta)
