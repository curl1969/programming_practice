def rem(num, k):
  if k == 0: return num
  if k == len(num): return "0"
  l,stack = [int(x) for x in num],[]
  for i in l:
    while k and stack and stack[-1] > i:
      k -= 1
      stack.pop()
    stack.append(i)
  if k > 0:
    stack = stack[:-k]
  ans = "".join([str(i) for i in stack])
  return str(int(ans))


num,k = "1432219",3
print(rem(num,k))
