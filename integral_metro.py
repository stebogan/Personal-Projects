import math
import random

N=100000
MEASURE=100

mean=0.0
meansq=0.0
delta=0.4
x=0

naccept=0

for meas in range(0,MEASURE):
  n=0
  mc=0.0
  x=random.random()
  for i in range(0,N):
    xtrial=x+delta*(random.random()-0.5)
    if ((xtrial>1.0) or (xtrial<0.0)):
      xtrial=1.0
    w=math.sqrt((1-xtrial*xtrial)/(1-x*x))

    if (random.random()<=w):
      x=xtrial
      naccept+=1
    mc+=math.pi/4*x


  mc = mc/N
  print("Metropolis Monte Carlo yields: %.16f " %mc)
  meansq+=mc*mc
  mean+=mc

meansq = meansq/MEASURE
mean = mean/MEASURE
naccept=naccept/(1.0*N*MEASURE)
print ("Metropolis acceptance: %.16f " %naccept)
error=math.sqrt((meansq-mean*mean)/MEASURE)
print("\n")
print("Monte Carlo mean: %.16f" %mean)
print("Monte Carlo error: %.16f" % error)
