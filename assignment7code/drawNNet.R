### by Chuck Anderson, for CS545
### http://www.cs.colostate.edu/~anderson/cs545
### You may use, but please credit the source.

drawNNet <- function(nnet,
                     inputNames = NULL,
                     outputNames = NULL,
                     gray=FALSE) {

  V <- nnet$V
  W <- nnet$W
  vmax <- max(abs(V))
  V <- V / vmax
  wmax <- max(abs(W))
  W <- W / wmax

  ni <- nrow(V) - 1
  nh <- ncol(V)
  no <- ncol(W)
  allw <- rbind( cbind(NA, V, NA), rep(NA,nh+2), cbind(t(W),NA))
  nr <- nrow(allw)
  nc <- ncol(allw)

  ## Set up plot
  plot(NA, NA, type = "n", xlim=c(0.5,nc+1.5), ylim=c(0.5,nr+0.5), axes=FALSE,
       asp=1, xlab="",ylab="")

  ## vertical lines
  xs <- 1:nh + 1
  y1 <- 0.5
  y2 <- nr+0.5
  segments(xs,y1,xs,y2)
  segments(1,0.5,1,no+0.5)
  ## constant 1 to output layer
  text(1,no+1,"1")
  ## horizontal lines
  ## hidden layer
  ys <- 1:(ni+1) + no + 1
  x1 <- 1
  x2 <- nh + 1.5
  segments(x1,ys,x2,ys)
  ## input labels
  if (is.null(inputNames))
    for (i in 1:ni) 
      text(x1-0.4,ys[ni+1 - i],bquote(x[.(i)]))
  else
    for (i in 1:ni) 
      text(x1-0.4,ys[ni+1 - i],inputNames[i])#,pos=2)
  text(x1-0.5,ys[ni+1],"1")
  ## output layer
  ys <- 1:no
  x2 <- nh + 2.5
  segments(x1,ys,x2,ys)
  ## output labels
  if (is.null(outputNames))
    for (i in 1:no) 
      text(x2+0.4,ys[no-i+1],bquote(y[.(i)]))
  else
    for (i in 1:ni) 
      text(x2+0.4,ys[no-i+1]-0.2,outputNames[i])
  ## cell bodies
  r <- 0.2
  hTri <- list(x=c(-r,r,0), y=c(r,r,-r))
  hTri$x <- hTri$x + 1
  hTri$y <- hTri$y + no + 1
  for (i in 1:nh) {
    hTri$x <- hTri$x + 1
    polygon(hTri,col="gray")
  }
  oTri <- list(x=c(-r,-r,r), y=c(-r,r,0))
  oTri$x <- oTri$x + nh + 2
  for (i in 1:no) {
    oTri$y <- oTri$y + 1
    polygon(oTri,col="gray")
  }

  ## Draw the weights
  allwc <- c(t(allw[nr:1,]))
  if (gray)
    colors <- ifelse(allwc < 0, "black", "gray")
  else
    colors <- ifelse(allwc < 0, "red", "green")

  symbols(expand.grid(1:nc,1:nr), squares=abs(allwc), bg=colors, #fg=NA,
          inches=FALSE, add=TRUE)

  ## Draw max values
  #text(nh+1.5,1+no+1+ni/2,bquote(v[max]==.(round(vmax,3))),pos=4)
  #text(1+(nh+1)/2,0.1,bquote(w[max]==.(round(wmax,3))),pos=3)
  mtext(bquote(v[max]==.(round(vmax,3))),3,1)
  mtext(bquote(w[max]==.(round(wmax,3))),1,1)
}

drawNNetDemo <- function() {
  
  ni <- 5
  nh <- 10
  no <- 2
  makew <- function(nr,nc) { matrix(1:(nr*nc) - nr*nc/2, nr,nc,byrow=TRUE)}
  makewr <- function(nr,nc) { matrix(runif(nr*nc)-0.5, nr,nc,byrow=TRUE)}
  net <- list(V = makew(ni+1,nh), W=makew(nh+1,no))

#  x11(type="Xlib")
  
  p <- par(mar=c(0,0,0,0))
  drawNNet(net)
  par(p)

  print("Press enter")
  scan()

  for (i in 1:100) {
    net <- list(V = makewr(ni+1,nh), W=makewr(nh+1,no))
    drawNNet(net)
    system("sleep 0.01")
  }
  
}
