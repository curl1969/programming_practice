
    # @param A : tuple of integers
    # @param B : integer
    # @return a list of integers
def twoSum(A, k):
  dict = {}
  for i,v in enumerate(A):
    if k-v in dict:
      return dict[k-v]+1,i+1
    elif v not in dict:
      dict[v] = i
  return []
print(twoSum([1,23,2,3,4],24))

