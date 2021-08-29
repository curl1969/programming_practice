       
def solve(n):
    Fib = [0, 1]
    for i in range(2, n + 1):
        Fib.append(Fib[i - 1] + Fib[i - 2])
    return Fib[n]


print(solve(3))  
