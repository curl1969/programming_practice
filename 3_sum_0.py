#class Solution:
    # @param A : list of integers
    # @return a list of list of integers
def threeSum(A):
    A.sort()
    k=0
    final=set()
    for i in range(len(A)-2):
        constant=A[i]
        start=i+1
        stop=len(A)-1
        while(stop>start):
            #if(abs((A[stop]+A[start]+constant)-k)<abs(final-k)):
            #   final=(A[start]+A[stop]+constant)
            
            if((A[stop]+A[start]+constant)>k):
                stop-=1
            elif((A[stop]+A[start]+constant)<k):
                start+=1
            else:
                final.add((constant,A[start],A[stop]))
                stop-=1
                start+=1
    print(list(final))
    return list(final)
    print(threeSum([1,2,3,4,5,6,-9]))
    print('hello world')
