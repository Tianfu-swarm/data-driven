import math

def S(n):
    result = 0
    for m in range(2, n + 1):
        term = math.comb(n, m) * math.factorial(m - 1) * math.factorial(n-1) / math.factorial(m-1)
        print(m, term)
        result += term
    return result

# 测试代码
n = 10  #节点数量
result = S(n)
print(result)

result_total = (n - 1) ** n

print(result_total)

print(result/result_total*100)
