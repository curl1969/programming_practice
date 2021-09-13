# a to z -> 1 to 26. Return the number of possible ways to decode a given string.
def dec_num(s):
    if len(s) == 0 or int(s[0])== 0: return 0
    n= len(s)
    res= [0]*(n+1)
    res[0]=1
    res[1]=1
    for i in range(1,n):
        v1,v2 = int(s[i:i+1]),int(s[i-1:i+1])
        if 0<v1 <=9: res[i+1]=res[i]
        if 10<=v2<=26: res[i+1]+= res[i-1]
        if res[i+1]== 0: return 0
        print(res)
            
    return res[n]
        
    
    
    
    
    
    

    






























#[1,1,2], i=1; v1=2,v2=12, i=2; 



print(dec_num('12023'))
