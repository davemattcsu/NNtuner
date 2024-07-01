######################################################################
### CS545, Fall 2009.  by Chuck Anderson
### Reinforcement learning using a neural network to learn a Q function.
### Neural network is trained using Moller's Scaled Conjugate Gradient algorithm.
### The agent's world is a point mass that moves in one dimension from integer positions 1 to 10.
### Goal is position 5.  The mass has no dynamics.  Actions are left, no move, and right one position.
######################################################################

source("nnQ.R")

######################################################################
### General RL code starts here for non-episodic tasks

########################################
### Return an action selected by epsilon-greedy policy
### Also return the associated Q value.

epsilonGreedy <- function(net,s,epsilon) {
  ## Forward pass through net to calculate Q values
  qs <- useNN(net,matrix(s,ncol=ni))
  if (runif(1) < epsilon) {
    ## random action
    a <- sample(actions,1)
    bestActIndex <- which(a==actions)
    Q <- qs[bestActIndex]
  } else {
    ## pick currently best action; a greedy choice
    bestActIndex <- which.max(qs)
    a <- actions[bestActIndex]
    Q <- qs[bestActIndex]
  }
  list(a=a,aI=bestActIndex,Q=Q)
}

########################################
### Return set of sample interactions.  All are matrices with one row per time step.

getSamples <- function(net,numSamples,epsilon) {
  X <- matrix(0,numSamples,ni)
  R <- matrix(0,numSamples,1)
  Q <- matrix(0,numSamples,1)
  Y <- matrix(0,numSamples,1)
  AI <- matrix(0,numSamples,1)

  ## Initial state
  s <- initialState()

  ## Select action using epsilon-greedy policy and get Q
  eg <- epsilonGreedy(net,s,epsilon)
  a <- eg$a
  aI <- eg$aI
  q <- eg$Q
  
  goal <- 5
  for (step in 1:numSamples) {

    ## Update state, s1 from s and a
    s1 <- nextState(s,a)

    ## Get resulting reinforcement
    r1 <- reinforcement(s,s1,goal)

    ## Select action for next step and get Q
    eg <- epsilonGreedy(net,s1,epsilon)
    a1 <- eg$a
    a1I <- eg$aI
    q1 <- eg$Q

    ## Collect
    X[step,] <- s
    R[step,1] <- r1
    Q[step,] <- q1
    Y[step,] <- q
    AI[step,1] <- aI

    ## Shift state, action and action index by one time step
    s <- s1
    a <- a1
    aI <- a1I

  }
  list(X=X, R=R, Q=Q, Y=Y, AI=AI)
}

### End of General RL code
######################################################################


######################################################################
### Start of code for 1-d dynamic point mass problem. Continues to end of file


########################################
### Return reinforcement given current state

reinforcement <- function(s,s1,goal) {
  if (abs(s1[1]-goal) < 2) {
    ## With 2 of goal, return 1
    1
  } else {
    ## Not within 2 of goal, return 0
    0
  }
}

########################################
### Return new random state

initialState <- function() {
  c(sample(10,1),0)
}

########################################
### Return updated state, given the action to apply to old state

nextState <- function(s,a) {
  ## s[1] is position, s[2] is velocity. a is -1, 0 or 1

  ## Euler integration time step
  deltaT <- 0.1

  ## Update position
  s[1] <- s[1] + deltaT * s[2]
  ## Update velocity. Includes friction
  s[2] <- s[2] + deltaT * (2 * a - 0.2 * s[2])

  ## Bound next position. If at limits, set velocity to 0.
  if (s[1] < 1) 
    s <- c(1,0)
  else if (s[1] > 10)
    s <- c(10,0)

  ## Return new state
  s
}

########################################
### Draw Q and policy as surfaces and contours or images

drawSurfaces <- function(net) {
  xs <- seq(1,10,len=20)
  xds <- seq(-4,4,len=20)
  q <- matrix(0,length(xs),length(xds))
  a <- matrix(0,length(xs),length(xds))
  nh <- ncol(net$V)
  z <- array(0,c(nh,length(xs),length(xds)))
  for (i in 1:length(xds)) {
    for (j in 1:length(xs)) {
        out <- nnOutput(net,matrix(c(xs[j],xds[i]),nrow=1))  ## use goal=5
        q[j,i] <- max(out$Y)
        a[j,i] <- actions[which.max(out$Y)]
        for (h in 1:nh)
          z[h,j,i] <- out$Z[h]
    }
  }
  cat("range of Q",range(q),"\n")
  ##Q
  persp(xs,xds,q,xlab="x",ylab="xdot",zlab="Q", phi=30, theta=70, ticktype="detailed")
  contour(xs,xds,q,xlab="x",ylab="xdot")
  ##policy
  persp(xs,xds,a,xlab="x",ylab="xdot",zlim=c(-1,1),zlab="Action", phi=30, theta=70, ticktype="detailed")
  image(xs,xds,a)
  ##hidden
  for (h in 1:nh) {
    persp(xs,xds,z[h,,],xlab="x",ylab="xdot",zlim=c(-1,1),zlab="Z", phi=30, theta=70, ticktype="detailed")
  }  
}

##################################################
### Main application code

x11(type="Xlib",width=10,height=10)

##############################
### Parameters for run, and initializations.

### Final probability of random action
finalEpsilon <- 0.001
### Hidden weight penalty.  0 means no penalty
lambda <- 0.00
### Discount factor
gamma <- 0.9

### Number of repetitions of generate-examples/update-net loop
nReps <- 100
### Number of interaction steps in each repetition
N <- rep(2000,nReps)
### Maximum number of iterations for SCG each repetition.
maxIterations <- 10

epsilonRate <- exp(log(finalEpsilon)/nReps)
cat("epsilon decay rate = ",epsilonRate,"\n")
### Initial epsilon is 1, for fully random action selection
epsilon <- 1

### Variables for plotting later
ftrace <- NULL
rtrace <- NULL
xtrace <- NULL
epsilontrace <- NULL

### Number of inputs to Q neural network,  position and velocity
ni <- 2  
### Number of hidden units.
nh <- 6
### Values of each of the 3 actions
actions <- c(-1,0,1)

### Set up graphics display
nplots <- 10 + nh
nr <- 4
nc <- ceiling(nplots/nr)
nempty <- nr*nc-nplots
pr <- par(mfcol=c(nr,nc))

##############################
### Create the neural network for Q function. It has ni (2) inputs, nh hidden units, and as many outputs as actions (3).
### When creating it, specify the range of each input so makeStandardizeF can create the correct function.
net <- makeNN(ni,nh,length(actions),matrix(c(1,10,-5,5),2,2))

##############################
### Main loop start
for (reps  in 1:nReps) {
  
  ## Collect N[reps] samples, each being s, a, r, s1, a1
  samples <- getSamples(net,N[reps],epsilon)

  ## Update the Q neural network.
  net <- updateNN(samples$X, samples$R, samples$Q, samples$Y, samples$AI,
                  net, gamma=gamma,lambda=lambda, fPrec=1e-8,nIter=maxIterations)

  ## Draw Q, policy, and hidden unit surfaces
  drawSurfaces(net)
  
  ## Draw neural network
  drawNNet(net,c("x","xd"))

  ## Track and plot trace of reinforcement values.
  rtrace <- c(rtrace,mean(samples$R))
  plot(rtrace,type="b")

  ## Track and plot epsilon values.
  epsilontrace <- c(epsilontrace,epsilon)
  plot(epsilontrace,type="b")
  
  ## Track and plot TD errors
  ftrace <- c(ftrace, net$ftrace)
  plot(ftrace,type="l")

  ## Plot last set of interactions, forming a trajectory
  plot(samples$X[,1],samples$X[,2],type="l",xlab="x",ylab="x dot",
       xlim=c(1,10),ylim=c(-5,5))
  
  ## Draw the policy for 0 velocity states.
  acts <- apply(useNN(net,cbind(matrix(1:10),0)),1,which.max) ## use goal 5
  x <- 1:10
  y <- rep(0,10)
  symbols(x,y,circles=rep(0.2,10),bg=acts,
          inches=FALSE,xaxt="n",yaxt="n",bty="n",xlab="",ylab="")
  text(x,y+0.4,c("L"," ","R")[acts],col=acts,cex=2,font=2)

  ## Advance to correct axes if necessary, for next repetition.
  if (nempty > 0)
    for (i in 1:nempty)
      plot(0,1,type="n",,xaxt="n",yaxt="n",ylab="",xlab="",bty="n")
    
  ## Update epsilon
  epsilon <- epsilonRate * epsilon

}

  
######################################################################
### Test the result from a number of starting positions
testIt <- function(steps) {
  x11(type="Xlib",width=10,height=10)
  xtraces <- NULL
  nTrips <- 50
  x0 <- seq(1,10,length=nTrips)
  for (trips in 1:nTrips) {
    s <- c(x0[trips],0) ## 0 velocity
    xtrace <- s
    for (step in 1:steps) {
      eg <- epsilonGreedy(net,s,epsilon=0)
      a <- eg$a
      s <- nextState(s,a)
      xtrace <- rbind(xtrace,s[1:2])
    }
    if (trips == 1)
      plot(xtrace[,1],xtrace[,2],type="l",xlim=c(0,11),ylim=c(-3,3))
    else
      lines(xtrace[,1],xtrace[,2],type="l",col=trips)
  } #trips
  points(5,0,col="purple",cex=3,pch=19)
  par(pr)
}

testIt(500)

      
      
