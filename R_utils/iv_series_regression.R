
###########################################
# From S. Wager 
# GRF repo 
# grf-labs/grf/experiments/baselines.R 
###########################################

################
# IV Series
################

library('splines')
library('AER')

iv.series = function(X, Y, W, Z, X.test, df = 3, interact = FALSE) {
  
  X.all = rbind(X, X.test)
  X.spl = Reduce(cbind , lapply(data.frame(X.all), function(xx) ns(xx, df)))
  
  cat(dim(X.all))
  cat(dim(X.spl))
                                
  if (interact) {
    X.reg = model.matrix( ~ . * . + 0, data = data.frame(X.spl))
  } else {
    X.reg = X.spl
  }
             
  if(ncol(X.reg) >= length(Y)) return(rep(NA, nrow(X.test)))
  
  series = ivreg(Y ~ X.reg[1:length(Y),] * W | X.reg[1:length(Y),] * Z)
                                
  capture.output(summary(series, diagnostics = TRUE), file = "summary_test.txt")

  beta = coef(series)
                                
  # NOTE: Calling this function on binary (e.g. one-hot encoded) Xs, so
  # filling NA values with 0.
  beta[is.na(beta)] = 0
  
  tau.hat = beta[ncol(X.reg) + 2] + X.reg[length(Y) + 1:nrow(X.test),] %*% beta[ncol(X.reg) + 2 + 1:ncol(X.reg)]
  
  tau.hat
  
}