def get_kth(l,h,k):
    def get_pow(n,table):
      if n in table: return table[n]
      if n== 1: return 0
      if not n%2: result = 1 + get_pow(n//2,table)
      else: result = 1 + get_pow(3*n+1,table)
      table[n] = result
      return result
    table = {}
    ls = [(l,get_pow(l,table)) for l in range(l,h+1)]
    print(ls)
    ls.sort(key = lambda x:(x[1],x[0]))
    return ls[k-1][0]
