######################################################################
### Neural networks for approximating Q function having discrete set of actions.
### Q for different actions are produced by different outputs. The number of outputs equals the number of possible actions.
### by Chuck Anderson, for CS545
### http://www.cs.colostate.edu/~anderson/cs545
### You may use, but please credit the source.

source("gradientDescents.R")
source("mlUtilities.R")
source("drawNNet.R")

######################################################################
### Given    | 1 4 2 |  and  c(3,1,2)
###          | 6 4 2 |
###          | 2 7 3 |
### return   | 2 |  and  c(3,1,2)
###          | 6 |
###          | 7 |

selectColumns <- function(M,columnIndices) {
  n <- length(columnIndices)
  z <- matrix(0,n,1)
  for (i in 1:n)
    z[i] <- M[i,columnIndices[i]]
  z
}

######################################################################
### Inverse of selectColumns

expandColumns <- function(vector,K,columnIndices) {
  errorMat <- matrix(0,length(vector),K)
  n <- length(columnIndices)
  for (i in 1:n)
    errorMat[i,columnIndices[i]] <- vector[i]
  errorMat
}

######################################################################
### Create a neural network

makeNN <- function(ni,nh,no,stateRanges) {
  ## stateRanges is 2 x ni, as if two extreme samples
  V <- matrix(0.1*(runif((ni+1)*nh)-0.5), ni+1,nh)
  W <- matrix(0.1*(runif((nh+1)*no)-0.5), nh+1,no)
  standardizeF <- makeStandardizeF(stateRanges)
  list(V=V, W=W, standardizeF=standardizeF, lambda=NULL)
}

######################################################################
### Update an existing neural network
### Use makeNN first.

updateNN <- function(X,R,Q,Y,AI, net, lambda=0,gamma=1,
                     xPrecision=1e-8,fPrecision=1e-8, nIterations=10000, rate=NULL) {
### rate non-null results in steepest() being used instead of scg()

  A <- makeIndicatorVars(AI)
  
  standardizeF <- net$standardizeF
  V <- net$V
  W <- net$W
  ni <- ncol(X)
  nh <- ncol(V)
  no <- ncol(W)

  ## Experimental code for growing the net by adding hidden units.
  ##   nhAdd <- 0
  ##   if (nhAdd > 0) {
  ##     V <- cbind(V, matrix(0.1*(runif((ni+1)*nhAdd)-0.5), ni+1,nhAdd))
  ##     W <- rbind(W, matrix(0.1*(runif((nhAdd)*no)-0.5), nhAdd,no))
  ##   }
  ##   nh <- nh + nhAdd

  X <- standardizeF(X)
  X1 <- cbind(1,X)

  nSamples <- nrow(X)

  pack <- function(v,w) {
    matrix(c(v,w))
  }

  unpack <- function(allw) {
    list(V = matrix(allw[1:((ni+1)*nh)],ni+1,nh),
         W = matrix(allw[-(1:((ni+1)*nh))],nh+1,no))
  }

  sqErrorF <- function(weights) {
    r <- unpack(weights)
    V <- r[[1]]
    W <- r[[2]]
    Z <- tanh(X1 %*% V)
    Y <- cbind(1,Z) %*% W
    ## Only calculate error for outputs corresponding to selected actions
    Ybest <- selectColumns(Y,AI)
    sqerror <- mean((R+gamma*Q-Ybest)^2) + lambda * sum(V[-1,]^2)
    0.5*sqerror
  }

  gradF <- function(weights) {
    r <- unpack(weights)
    V <- r[[1]]
    W <- r[[2]]
    Z <- tanh(X1 %*% V)
    Y <- cbind(1,Z) %*% W
    ## Only calculate gradient with respect to error for outputs corresponding to selected actions.
    Ybest <- selectColumns(Y,AI)
    error <- R + gamma*Q - Ybest
    errorMat <- expandColumns(error, ncol(Y), AI)
    dV <- -t(X1) %*% (errorMat %*% t(W[-1,,drop=FALSE]) * (1-Z^2)) / nSamples +
      lambda * rbind(0,V[-1,,drop=FALSE])
    dW <- -t(cbind(1,Z)) %*% errorMat / nSamples ##+ lambda * W
    pack(dV,dW)
  }

  if (is.null(rate)) {
    scgresult <- scg(pack(V,W),sqErrorF,gradF, ftracep=TRUE,xtracep=TRUE,
                     xPrecision = xPrecision, fPrecision = fPrecision,
                    nIterations = nIterations)
  } else {
    scgresult <- steepest(pack(V,W),sqErrorF,gradF, ftracep=TRUE,xtracep=TRUE,
                          xPrecision = xPrecision, fPrecision = fPrecision,
                          nIterations = nIterations, stepsize=rate)
  }

  cat("Update stopped due to limit on ",scgresult$reason,"\n")
  r <-  unpack(scgresult$x)
  V <- r[[1]]
  W <- r[[2]]

  list(standardizeF=standardizeF, V=V, W=W, lambda=lambda,
       ftrace=scgresult$ftrace,xtrace=scgresult$xtrace)
}

useNN <- function(nnet,X) {
  if (!is.null(nnet$standardizeF))
    X <- nnet$standardizeF(X)
  X1 <- cbind(1,X)
  Z <- tanh(X1 %*% nnet$V)
  Y <- cbind(1,Z) %*% nnet$W
  Y
}

######################################################################
### Like useNN, but also returns hidden unit outputs.

nnOutput <- function(nnet,X) {
  if (!is.null(nnet$standardizeF))
    X <- nnet$standardizeF(X)
  Z <- tanh(cbind(1,X) %*% nnet$V)
  Y <- cbind(1,Z) %*% nnet$W
  list(Z=Z, Y=Y)
}
  
