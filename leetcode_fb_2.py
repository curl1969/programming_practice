#union find



#921. Minimum Add to Make Parentheses Valid: Using a stack is better than trying to solve it using just a O(1) space. Every time we see a ")" , we pop (and set ans += 1 if we can't pop from an empty stack) from the stack or we add "(" into the stack. The final answer is ans+len(Stack) 

class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        stack = []
        ans = 0
        for elem in s:
            if elem == ')' and stack == []: ans += 1 
            elif elem == '(': stack.append(elem)
            else: stack.pop()
        return ans+ len(stack)
        
#1973. Count Nodes Equal to Sum of Descendants: recursion , at leaf (left+right+node.val) 

class Solution:
    def equalToDescendants(self, root: Optional[TreeNode]) -> int:
        result = 0
        def func(node):
            nonlocal result
            if not node: return 0
            
            left = func(node.left)
            right = func(node.right)
            
            if node.val == left + right:
                result += 1
            
            return left + right + node.val
            
        func(root)
        return result
#1104. Path In Zigzag Labelled Binary Tree       
from math import log
class Solution:
    def pathInZigZagTree(self, label: int) -> List[int]:
        val,result = int(log(label, 2)), [label]
        while val:
            diff = result[-1] - 2**val
            result.append(result[-1]- diff - diff//2-1)
            val -= 1
        return result[::-1]
        
class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        count = 0
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] == "X":
                    if r and board[r-1][c] == "X": continue
                    if c and board[r][c-1] == "X": continue
                    count += 1
        return count
        
# 1522: Once we get the max and 2nd max depth from any node to the leaves then we can find the longest path through that node by 
# sum of the dist and subtract the depth of the Node from where the recursion is called. We found the max and 2nd max in one pass starting
# from a Node. Everytime we find them we update the diameter.
class Solution:
    def diameter(self, root: 'Node') -> int:
        diameter = 0
        
        def helper(node,curr_depth):
            nonlocal diameter
            if len(node.children) == 0: return curr_depth
            l1,l2 = curr_depth,0
            for child in node.children:
                depth = helper(child,curr_depth+1)
                if depth > l1:
                    l1,l2 = depth,l1
                elif depth > l2 and depth <= l1:
                    l2 = depth
                    
            dist = (l1+l2) - 2*curr_depth
            diameter = max(diameter,dist)
            return l1
        
        helper(root,0)
        return diameter
        
# Permutation
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        if not nums : return [[]]
        return [[nums[i]] + j for i in range(len(nums)) for j in self.permute(nums[:i]+nums[i+1:])]
 #Pruning BIn tree       
class Solution:
    def pruneTree(self, root: TreeNode) -> TreeNode:
        if not root: return None
        left = self.pruneTree(node.left)
        root.left = left
        right = self.pruneTree(node.right)
        root.right = right
        if (root.val == 0 and root.left == None and root.right == None): return None
        return root
# Intersection of list of lists : Main idea : among two intervals, the one with largest end point has a chance to intersect with other intervals. We should remove the interval with smaller endpoint and run a two pointer accordingly  

class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        result = []
        i,j = 0,0
        while i< len(firstList) and j<len(secondList):
            beshi = min (firstList[i][1],secondList[j][1])
            kom = max (firstList[i][0], secondList[j][0])
            if kom <= beshi:
                result.append([kom,beshi])
            if firstList[i][1] > secondList[j][1]:
                j += 1
            else:
                i += 1

        return result
# List all possible subsets        
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        if not nums: return [[]]
        ans = [[]]
        for num in nums:
            ans += [x + [num] for x in ans]
        return ans
 #Queue Reconstruction by Height      
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        
        people.sort(key = lambda x: (-x[0],x[1]))
        result = []
        for p in people:
            result.insert(p[1],p)
        return result
#22.Generate Parentheses 
def recur(result,s,op,cl,n):
    if op == n and cl == n:
        result.append(s)
        return
    if op < n:
        recur(result,s+"(",op+1,cl,n)
    if cl < op:
        recur(result,s+")",op,cl+1,n)

#1110. Delete Nodes And Return Forest
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result = []
        recur(result,"",0,0,n)
        return result
        
class Solution:
    def delNodes(self, root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
        ans = []
        to_delete = set(to_delete)
        
        def helper(node):
            if not node:
                return None
            node.left = helper(node.left)
            node.right = helper(node.right)
			
			# add children of a node that is to be deleted
            if node.val in to_delete:
                if node.left: 
                    ans.append(node.left)
                if node.right:
                    ans.append(node.right)
                return None
            return node
                
        helper(root)
		# if root is not to be deleted then add it
        if root.val not in to_delete:
            ans.append(root)
        return ans
#695. Max Area of Island      
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        seen = set()
        def dfs(r,c):
            
            if not (0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 1 and (r,c) not in seen): return 0
            seen.add((r,c))
            return (dfs(r,c+1) + dfs(r,c-1) + dfs(r-1,c) + dfs(r+1,c) + 1)
        
        return max(dfs(r,c) for r in range(len(grid)) for c in range(len(grid[0])))

#791. Custom Sort String
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        d = collections.Counter(s)
        l = []
        for i in order:
            l.append(i*d[i])
            d[i] = 0
            
        for elem in d:
            l.append(elem*d[elem])
        return "".join(l)

#451. Sort characters by frequency 
        
class Solution:
    def frequencySort(self, s: str) -> str:
        d = collections.Counter(s)
        l = [key*val for key,val in d.most_common()] # d.most_common() = (key,val)
        return "".join(l)

# Bucket sort can be done on a dict in linear time




#1244. Design A Leaderboard       
class Leaderboard:

    def __init__(self):
        self.lead = defaultdict()

    def addScore(self, playerId: int, score: int) -> None:
        if playerId not in self.lead:
            self.lead[playerId] = 0
        self.lead[playerId] += score

    def top(self, K: int) -> int:
        vals = [x for _, x in sorted(self.lead.items(), key=lambda item: item[1])]
        vals.sort(reverse = True)
        total,i  = 0,0
        while i < K:
            total += vals[i]
            i += 1
        return total

    def reset(self, playerId: int) -> None:
        self.lead[playerId] = 0

#739. Daily Temperatures
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        l = len(temperatures)
        res = [0]*l
        stack = []
        for curridx, currtemp in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < currtemp:
                lastdayidx = stack.pop()
                res[lastdayidx] = curridx - lastdayidx
            stack.append(curridx)
        return res
        
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        
        def getheight(root):# returns ht of bin TreeNode
            if root.left == None and root.right === None: return 0
            left,rigth = 0,0
            if root.left != None:
                left = getheight(root.left)
            if root.right != None:
                right = getheight(root.right)
            return max(left,right) + 1
        
        SUM = []
        def levelsum(node,level):
            global SUM
            if not node: return
        SUM[level] += node.val
        
        levelsum(node.left,level+1)
        levelsum(node.right,level+1)
        
    levels = getheight(root)+1
    SUM = [0]*levels
    levelsum(root,1)
    
    return SUM.indexof(max(SUM))#
# 951. Flip Equivalent Binary Trees    
class Solution:
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        if root1 == None and root2 == None: return True
        if root1 == None and root2 != None: return False
        if root1 != None and root2 == None: return False
        if root1.val == root2.val:
            return (self.flipEquiv(root1.left,root2.left) and self.flipEquiv(root1.right,root2.right)) or (self.flipEquiv(root1.left,root2.right) and self.flipEquiv(root1.right,root2.left))
    
class Solution:
    def minSwaps(self, s: str) -> int:
        stack = []
        ans = 0
        for i in s:
            if i == '[': stack.append(i)
            if i == "]":
                if not stack:
                    ans +=1
                    stack.append(i)
                else:
                    stack.pop()
        return ans
        
# return the maximum product from a given array
        def prod_list(l): #given a list, returns the product of the elements
            res = 1
            for elem in l:
                res = res * elem
            return res
        def pos_part(l): #returns only the positive lements of the list
            ans = []
            for elem in l:
                if elem > 0:
                    ans.append(elem)
            return ans
        def neg_part(l): #returns only the negative lements of the list
            ans = []
            for elem in l:
                if elem < 0:
                    ans.append(elem)
            return ans
        def check(l): #return if our list contains only zero and negative numbers
            for elem in l:
                if elem >0:
                    return False
            return True
        def max_elem(l): #returns the minimum element of a given list
            curr_max = l[0]
            for elem in l:
                curr_max = max (curr_max,elem)
            return curr_max
        def max_prod(l):
            if l == [0]: return 0
            if check(l) == True and len(l)== 2: return 0
            curr_prod =1
            n = prod_list(pos_part(l))
            l2 = neg_part(l)
            if len(l2)%2 == 0:
                return n*prod_list(l2)
            if len(l2)%2 == 1:
                M = n*prod_list(l2)
                return M //max_elem(l2)
                
                
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        
        def split_distances(self, remaining: List[int], distances: List[float],mid: int) -> List[List[int]]:
# puts the dist closer than mid in the list closer and same way for farther
        closer, farther = [], []
        for index in remaining:
            if distances[index] <= mid:
                closer.append(index)
            else:
                farther.append(index)
        return [closer, farther]

    def euclidean_distance(self, point: List[int]) -> float: #return euclidean dist
        return point[0] ** 2 + point[1] ** 2
        
        
        distances = [self.euclidean_distance(point) for point in points]# list of the dist of all points from origin
        remaining = [i for i in range(len(points))]# list of the indices
        low, high = min(distances), max(distances)# the range of Bin search is created
        

        closest = []
        # Bin search begins
        while k:
            mid = (low + high) / 2
            closer, farther = self.split_distances(remaining, distances, mid)
            if len(closer) > k:
                # If more than k points are in the closer distances
                # then discard the farther points and continue
                remaining = closer
                high = mid
                # in this case we dont reduce k since all of the points reqd are in closer and dont know yet how many to pick
            else:
                # Add the closer points to the answer array and keep
                # searching the farther distances for the remaining points
                k -= len(closer)
                closest.extend(closer)
                remaining = farther
                low = mid
                
        # Return the k closest points using the reference indices
        return [points[i] for i in closest]
#48. Rotate Image
def transpose(matrix):
    for i in range(len(matrix)):
        for j in range(i+1,len(matrix)):
            matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
        
def reflect(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix)//2):
            matrix[i][j],matrix[i][-j-1] = matrix[i][-j-1],matrix[i][j]

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:    
        transpose(matrix)
        reflect(matrix)
#------------------------------------------------------------------------------------------------------------------------------------   
#K-th smallest element in binary search tree
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder(node):
            if not node: return []
            return inorder(node.left) + [node.val] + inorder(node.right)
        return inorder(root)[k-1]
#--------------------------------------------------------------------------------------------------------------------------------        
#311. Sparse Matrix Multiplication
class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        mydict = {}
        for i in range(len(mat1)):
            for j in range(len(mat1[0])):
                if mat1[i][j] != 0:
                    if i not in mydict:
                        mydict[i] = [j]
                    else:
                        mydict[i].append(j)
                        
        # mydict now contain all rows as keys and vals for each keys are the indices where elems are nonzero
        
        ans = [[0] * len(mat2[0]) for _ in range(len(mat1))]
        for key in mydict:
            for colidx in range(len(mat2[0])):
                row = mydict[key]
                col = [mat2[i][colidx] for i in range(len(mat2))]
                for x in row:
                    ans[key][colidx] += mat1[key][x]*mat2[x][colidx]
        return ans
#---------------------------------------------------------------------------------------------------------------------------- #1721. Swapping Nodes in a Linked List     
def findlength(head):
    cur=head
    count=0
    while(cur!=None):
        count=count+1
        cur=cur.next
    return count
#to convert linked list to array
def convertarr(head):
    len=findlength(head)
    arr=[]
    index=0
    cur=head
    
    while(cur!=None):
        arr.append(cur.val)
        cur=cur.next
    return arr
#to convert array to linked list 
def lst2link(lst):
    cur = dummy = ListNode(0)
    for e in lst:
        cur.next = ListNode(e)
        cur = cur.next
    return dummy.next


class Solution:
    def swapNodes(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        n = findlength(head)
        if n == 1: return head
        
        # the kth node from the back will be n-k th node from the front since we start from 0
        # so the kth node is actually the k-1 th node in this sense
        ourlist = convertarr(head)
        ourlist[k-1],ourlist[n-k] = ourlist[n-k],ourlist[k-1]
        ans = lst2link(ourlist)
        return ans
        
