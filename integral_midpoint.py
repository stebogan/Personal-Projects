import math

sum = 0
#Lower bound
min = 0
# Upper bound
max = (math.pi)

dx = 0.00001

# Integrand Function
def f(x):
    return math.sin(x)

x = min
while(x <= max):
    sum = sum + dx*f(x+0.5*dx)
    x = x+dx


print("Newtonian integration yields: %.16f" %sum)
print("Error: %.16f"%(abs(2-sum)))
