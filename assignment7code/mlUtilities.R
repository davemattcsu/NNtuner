### by Chuck Anderson, for CS545
### http://www.cs.colostate.edu/~anderson/cs545
### You may use, but please credit the source.

#source("gradientDescents.R")

######################################################################
### General functions


makeStandardizeF <- function(X) {
  if (missing(X)) {
    cat("Usage:
         standardize <- makeStandardizeF(X)  ## X is nSamples x nDimensions
         Xs <- standardize(X)
         X2s <- standardize(X2)\n")
    return(invisible())
  }
  ## X is nSamples x nDimensions
  mu <- colMeans(X)
  sigma <- sd(X) ##sd should be named colSds

  function(newX) {
    nr <- nrow(newX)
    nc <- ncol(newX)
    (newX - matrix(mu,nr,nc,byrow=TRUE)) / matrix(sigma,nr,nc,byrow=TRUE)
  }
}

makeDeStandardizeF <- function(T) {
  ## T is nSamples x nDimensions
  mu <- colMeans(T)
  sigma <- sd(T) ##sd should be named colSds

  function(newT) {
    nr <- nrow(newT)
    nc <- ncol(newT)
    newT * matrix(sigma,nr,nc,byrow=TRUE) + matrix(mu,nr,nc,byrow=TRUE)
  }
}

makeIndicatorVars <- function(Y) {
  if (!is.matrix(Y))
    Y <- matrix(Y)
  classes <- sort(unique(Y))
  N <- nrow(Y)
  K <- length(classes)
  logicalMatrix <- (matrix(Y,N,K) == matrix(classes,N,K,byrow=TRUE))
  mode(logicalMatrix) <- "numeric" ## to convert to numbers 0, 1
  logicalMatrix
}


### combine(results,parameterColumns,valueColumns,func)
combine <- function(results,parameterColumns,valueColumns,func=colMeans) {
  uniqueCombos <- unique(results[,parameterColumns,drop=FALSE])
  comboResult <- c()
  for (uci in 1:nrow(uniqueCombos)) {
    parameters <- uniqueCombos[uci,]
    mask <- apply(results[,parameterColumns,drop=FALSE], 1,
                  function(ps) all(ps==parameters))
    comboResult <- rbind(comboResult,
                         c(parameters,
                           func(results[mask,valueColumns,drop=FALSE])))
  }
  comboResult
}


