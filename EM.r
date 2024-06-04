#################################
####### EM Algorithm ############
#################################

# This file contains two examples of implementing the EM algorithm. The first 
# example is a mixture model of two binomial distributions where we augment
# the data with a latent variable that denotes from which distribution the data
# comes from. The second example is a missing data problem with Poisson regression.
# This is an exercise from Casella and Berger

################################################################################
# EM for Mixture Model Classification ##########################################
################################################################################

# Initializing the Algorithm
n<-1000
P.init<-c(.1, .6, .7)

# True Values
m<-20
q<-.4
p1<-.3
p2<-.9
P.true<-c(q, p1, p2)

# Generate Data
dat<-function(n){
  Z<-rbinom(n, 1, q)
  X<-c()
  for(i in 1:n){
    if(Z[i]==1){
      Xi<-rbinom(1,m,p1)
    }
    else{
      Xi<-rbinom(1,m,p2)
    }
    X<-c(X, Xi)
  }
  
  return(X)
}

# Define variable to calculate posterior probabilities
gamma1<-function(X, P){
  gam<-c()
  for(i in 1:n){
    num<-P[1]*dbinom(X[i], m, P[2])
    denom<-P[1]*dbinom(X[i], m, P[2])+(1-P[1])*dbinom(X[i], m, P[3])
    g<-num/denom
    gam<-c(gam,g)
  }
  return(gam)
}

# EM Algorithm Function
EM<-function(data, P, epsilon=.00001){
  q_vec<-c()
  p1_vec<-c()
  p2_vec<-c()
  P.old<-P
  
  gamma1_n<-gamma1(data, P.old)
  gamma0_n<-1-gamma1(data, P.old)
  
  q_n<-sum(gamma1_n)/sum(gamma0_n+gamma1_n)
  p1_n<-sum(gamma1_n*data)/sum(m*gamma1_n)
  p2_n<-sum(gamma0_n*data)/sum(m*gamma0_n)
  
  q_vec<-c(q_vec, q_n)
  p1_vec<-c(p1_vec, p1_n)
  p2_vec<-c(p2_vec, p2_n)
  
  P.new<-c(q_n, p1_n, p2_n)
  
  count<-0
  while(sum((P.new-P.old)^2)>=epsilon){
    P.old<-P.new
    
    gamma1_n<-gamma1(data, P.old)
    gamma0_n<-1-gamma1(data, P.old)
    
    q_n<-sum(gamma1_n)/sum(gamma0_n+gamma1_n)
    p1_n<-sum(gamma1_n*data)/sum(m*gamma1_n)
    p2_n<-sum(gamma0_n*data)/sum(m*gamma0_n)
    
    q_vec<-c(q_vec, q_n)
    p1_vec<-c(p1_vec, p1_n)
    p2_vec<-c(p2_vec, p2_n)
    
    P.new<-c(q_n, p1_n, p2_n)
    
    count<-count+1
    if(count%%100==0){
      print(count)
    }
    if(count>5000){
      break
    }
    
  }
  return(list(est=P.new, q_trace=q_vec, p1_trace=p1_vec, p2_trace=p2_vec, iterations=count))
}

data<-dat(n)
res<-EM(data, P.init)

# PLotting the data as well as the trace of our parameters
par(mfrow=c(2,2))
hist(data)
plot(res$q_trace)
plot(res$p1_trace)
plot(res$p2_trace)




################################################################################
# EM for Missing Poisson Data ##################################################
################################################################################

# Initialize the true values
n<-100
m<-200
beta<-5
tau<-rep(1/n, n)
vrai<-c(tau, beta)

# Generate the Data
Y<-c()
for(i in 1:n){
  y<-rpois(1, m*beta*tau[i])
  Y<-c(Y,y)
}

X<-rmultinom(1, m, tau)

# Initialize Starting Guesses for Parameters
t<-c(rep(1/(2*n), n/2 ), rep(3/(2*n), (n/2)))
b<-3
init<-c(t,b)

# EM algorithm Function
EM<-function(X, Y, init, epsilon){
  vec<-init
  t<-vec[1:n]
  b<-vec[n+1]
  k<-0
  while(sum((vrai-vec)^2)>=epsilon){
    b0 <- b
    t0 <- t
    
    b<-sum(Y)/(m*t0[1]+sum(X[2:n]))
    
    t[1]<-(Y[1]+m*t0[1])/(m*b)
    
    for(i in 2:n){
      t[i]<-(Y[i]+X[i])/(m*b)
    }
    
    
    vec<-c(t,b)
    
    
    k<-k+1
    if(k%%50000==0){
      print(k)
    }
    
    if(k>500000){
      break
    }
    
  }
  return(list(sol=vec, iter=k, error_tau=sum((vrai[1:n]-vec[1:n])^2), error_beta=sum((vrai[n+1]-vec[n+1])^2)))
}


res<-EM(X, Y, init, .00001)
