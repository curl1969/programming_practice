class Solution:
    # @param A : list of integers
    # @return a list of integers
    def plusOne(self, A):
      while len(A) > 0 and A[0] == 0:
        del A[0]
      A.insert(0,0)
      carry = 1
      for i in range(len(A)-1,-1,-1):
        result = A[i] + carry
        carry = result // 10
        A[i] = result % 10
      if A[0] == 0:
        del A[0]
      return A
