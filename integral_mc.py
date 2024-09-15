import math
import random

# Number of MC samples
M = int(input("number of mc samples?"))
# Number of iterations per sample
N=1000
# Upper integral bound
b = 1
# Lower integral bound
a = -1

# Integrand function
def f(x):
  return math.sqrt(1-x*x)

mean = 0.0
meansq = 0.0

for i in range(0,M):
  mc = 0.0

  for j in range(0,N):
    x = random.uniform(a,b)
    mc = mc + f(x)

  mc = mc * (b-a)/N
  print("Monte Carlo yields: %.16f" %mc)
  meansq += mc*mc
  mean += mc

meansq = meansq / M
mean = mean / M
error = math.sqrt((meansq - mean*mean)/M)

print("\n")
print("Monte Carlo mean: %.16f" %mean)
print("Error: %.16f" %error)
