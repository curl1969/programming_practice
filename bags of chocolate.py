from heapq import *

class Solution:
    # @param A : integer
    # @param B : list of integers
    # @return an integer
    def nchoc(self, A, B):
        h = [-b for b in B]
        heapify(h)
        s = 0
        for _ in range(A):
            chocs = -heappop(h)
            s = (s + chocs) % 1000000007
            heappush(h, -(chocs // 2))
        return s   
      
      
    
      
  
         
      
        
      

        
        
