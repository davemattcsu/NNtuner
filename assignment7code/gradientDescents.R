### by Chuck Anderson, for CS545
### http://www.cs.colostate.edu/~anderson/cs545
### You may use, but please credit the source.

######################################################################
### Steepest descent

steepest <- function(x,
                     f,gradf,
                     ...,
                     stepsize=0.1,
                     evalFunc=function(x) {paste("Eval",x)},
                     nIterations=1000,
                     xPrecision=0.001*mean(x),
                     fPrecision=0.001*mean(f(x,...)),
                     xtracep=FALSE,
                     ftracep=FALSE) {

### Example.  This is printed when called like   steepest()
  if (missing(x)) {
    cat("Example:
  parabola <- function(x,xmin,s) {
     d <- x - xmin
     t(d) %*% s %*% d 
  }
  parabolaGrad <- function(x,xmin,s) {
     d <- x - xmin
     2 * s %*% d 
  }
  center <- c(5,5)
  S <- matrix(c(5,4,4,5),2,2)
  firstx <- c(-1.0,2.0)
  r <- steepest(firstx, parabola, parabolaGrad, center, S, stepsize=0.01,xPrecision=0.001, nIterations=1000)
  cat(\"Optimal: point\",r$x,\"f\",r$f,\"\\n\") \n")
  return()
  }

  i <- 1
  xtrace <- NULL
  if (xtracep) 
    xtrace <- c(x)
  oldf <- f(x,...)
  ftrace <- NULL
  if (ftracep)
    ftrace <- c(oldf)
  
  while (i < nIterations) {
    g <- gradf(x,...)
    newx <- x - stepsize * g
    newf <- f(newx,...)
    if (i %% (nIterations/10) == 0)
      cat("Steepest: Iteration",i,"Error",evalFunc(newf),"\n")
    if (xtracep)
      xtrace <- cbind(xtrace,newx)
    if (ftracep)
      ftrace <- c(ftrace,newf)
    if (any(is.nan(newx)) || is.nan(newf))
      stop("Error: Steepest descent produced newx that is NaN. Stepsize may be too large.")
    if (any(is.infinite(newx)) || is.infinite(newf))
      stop("Error: Steepest descent produced newx that is NaN. Stepsize may be too large.")
    if (max(abs(newx - x)) < xPrecision)
      return (list(x=newx, f=newf, xtrace=xtrace, ftrace=ftrace, reason="x precision"))
    if (abs(newf - oldf) < fPrecision)
      return (list(x=newx, f=newf, xtrace=xtrace, ftrace=ftrace, reason="f precision"))
    x <- newx
    oldf <- newf
    i <- i + 1
  }
  return (list(x=newx, f=newf, xtrace=xtrace, ftrace=ftrace, reason="did not converge"))
}


######################################################################
### Scaled Conjugate Gradient algorithm from
###  "A Scaled Conjugate Gradient Algorithm for Fast Supervised Learning"
###  by Martin F. Moller
###  Neural Networks, vol. 6, pp. 525-533, 1993
###
###  Adapted by Chuck Anderson from the Matlab implementation by Nabney
###   as part of the netlab library.
###
###  Call as   scg()  to see example use.

scg <- function(x, 
                f,gradf,
                ...,
                evalFunc=function(x) {paste("Eval",x)},
                nIterations=1000,
                xPrecision=0.001*mean(x),
                fPrecision=0.001*mean(f(x,...)),
                xtracep=FALSE,
                ftracep=FALSE) {

### Example.  This is printed when called like   scg()
  
  if (missing(x)) {
    cat("Example:
  parabola <- function(x,xmin,s) {
     d <- x - xmin
     t(d) %*% s %*% d 
  }
  parabolaGrad <- function(x,xmin,s) {
     d <- x - xmin
     2 * s %*% d 
  }
  center <- c(5,5)
  S <- matrix(c(5,4,4,5),2,2)
  firstx <- c(-1.0,2.0)
  r <- scg(firstx, parabola, parabolaGrad, center, S)
  cat(\"Optimal: point\",r$x,\"f\",r$f,\"\\n\") \n")
  return()
  }
  
### from Nabney's netlab matlab library
  
  nvars <- length(x)
  sigma0 <- 1.0e-6
  fold <- f(x, ...)
  fnow <- fold
  gradnew <- gradf(x, ...)
  gradold <- gradnew
  d <- -gradnew				# Initial search direction.
  success <- 1				# Force calculation of directional derivs.
  nsuccess <- 0				# nsuccess counts number of successes.
  beta <- 1.0				# Initial scale parameter.
  betamin <- 1.0e-15 			# Lower bound on scale.
  ##betamax <- 1.0e1			# Upper bound on scale.
  betamax <- 1.0e20			# Upper bound on scale.
  j <- 1				# j counts number of iterations.

  xtrace <- NULL
  if (xtracep)
    xtrace <- c(x)
  ftrace <- NULL
  if (ftracep)
    ftrace <- c(fnow)
  
### Main optimization loop.
  while (j <= nIterations) {

    ## Calculate first and second directional derivatives.
    if (success == 1) {
      mu <- t(d) %*% gradnew
if (is.nan(mu)) browser()
      if (mu >= 0) {
        d <- - gradnew
        mu <- t(d) %*% gradnew
      }
      kappa <- t(d) %*% d
      if (kappa < .Machine$double.eps)
        return(list(x=x, f=fnow, xtrace=xtrace, ftrace=ftrace,reason="machine precision"))
      sigma <- sigma0/sqrt(as.numeric(kappa))
      xplus <- x + sigma * d
      gplus <- gradf(xplus, ...)
      theta <- (t(d) %*% (gplus - gradnew))/sigma
    }

    ## Increase effective curvature and evaluate step size alpha.
    delta <- theta + beta * kappa
if (is.nan(delta)) browser()
    if (delta <= 0) {
      delta <- beta * kappa
      beta <- beta - theta/kappa
    }
    alpha <- - as.numeric(mu/delta)
    
    ## Calculate the comparison ratio.
    xnew <- x + alpha * d
    fnew <- f(xnew, ...)
    Delta <- 2 * (fnew - fold) / (alpha*mu)
    if (!is.nan(Delta) && Delta  >= 0) {
      success <- 1
      nsuccess <- nsuccess + 1
      x <- xnew
      if (xtracep)
        xtrace <- cbind(xtrace, x)
      if (ftracep)
        ftrace <- c(ftrace,fnew)
      fnow <- fnew
    } else {
      success <- 0
      fnow <- fold
    }

    if ((j %% (nIterations/10)) == 0)
      cat("SCG: Iteration",j,"fValue",evalFunc(fnow),"Scale",beta,"\n")


    if (success == 1) {
      ## Test for termination

      ##print(c(max(abs(alpha*d)),max(abs(fnew-fold))))
      
      if (max(abs(alpha*d)) < xPrecision)
        return(list(x=x, f=fnow, xtrace=xtrace, ftrace=ftrace, reason="x Precision"))
      else if (max(abs(fnew-fold)) < fPrecision)
        return(list(x=x, f=fnow, xtrace=xtrace, ftrace=ftrace, reason="f Precision"))
      else {
        ## Update variables for new position
        fold <- fnew
        gradold <- gradnew
        gradnew <- gradf(x, ...)
        ## If the gradient is zero then we are done.
        if (t(gradnew) %*% gradnew == 0) {
          return(list(x=x, f=fnow, xtrace=xtrace, ftrace=ftrace, reason="gradient zero"))
        }
      }
    }

    ## Adjust beta according to comparison ratio.
    if (is.nan(Delta) ||  Delta < 0.25)
      beta <- min(4.0*beta, betamax)
    else if (Delta > 0.75)
      beta <- max(0.5*beta, betamin)

    ## Update search direction using Polak-Ribiere formula, or re-start 
    ## in direction of negative gradient after nparams steps.
    if (nsuccess == nvars) {
      d <- -gradnew
      nsuccess <- 0
    } else {
      if (success == 1) {
        gamma <- t(gradold - gradnew) %*% gradnew/mu
        d <- as.numeric(gamma) * d - gradnew
      }
    }
    j <- j + 1
  }

  ## If we get here, then we haven't terminated in the given number of 
  ## iterations.

  print("Did not converge.")
  return(list(x=x, f=fnow, xtrace=xtrace, ftrace=ftrace, reason="did not converge"))  
}


