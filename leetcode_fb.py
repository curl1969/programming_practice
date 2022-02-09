#finding a circle in a linked list
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head: return False
        slow = head
        fast = head.next
        while slow != fast:
            if fast is None or fast.next is None:
                return False
            slow = slow.next
            fast = fast.next.next
        return True
#finding majority element in a sequence
 class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        candidate = None
        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)
        return candidate
#balanced binary tree 
def height(root):
    if root == None: return -1
    else:
        return 1 + max (height(root.left),height(root.right))
    
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if not root: return True
        return abs(height(root.left)-height(root.right)) < 2 and self.isBalanced(root.left) and self.isBalanced(root.right)

#number of new flowers 
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        count = 0
        for i in range(len(flowerbed)):
            if flowerbed[i] == 0 and (i==0 or flowerbed[i-1] == 0) and (i == len(flowerbed)-1 or flowerbed[i+1] == 0):
                flowerbed[i] = 1
                count += 1
        return count >= n
        
#Path sum on a Tree 
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root: return False
        targetSum -= root.val
        if root.left == None and root.right ==  None and targetSum == 0: return True
        return self.hasPathSum(root.left, targetSum) or self.hasPathSum(root.right, targetSum)
        
#Dot product of two sparse vectors
class SparseVector:
    def __init__(self, nums: List[int]):
        self.array = nums

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        result = 0
        for num1,num2 in zip(self.array,vec.array):
            result += num1*num2
        return result

# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)
#buildings with ocan view
class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        curr_max = -1
        ocean_view = []
        for i in range(len(heights)-1,-1,-1):
            if heights[i] > curr_max:
                curr_max = max(curr_max,heights[i])
                ocean_view.append(i)
      
        return ocean_view[::-1]
#Binary Search Tree to Greater Sum Tree
class Solution:
    def bstToGst(self, root: TreeNode) -> TreeNode:
        
        total = 0
        
        def check(node):
            
            nonlocal total
            
            if not node:
                return 0
            
            check(node.right)
            total += node.val
            node.val = total
            check(node.left)
        
        check(root)
        
        return root
#Nested list weight sum 
class Solution:    
    def depthSum(self, nestedList):
        return sum([self.getSum(e, 1) for e in nestedList])
        
    def getSum(self, elem, depth):
        if elem.isInteger():
            return elem.getInteger() * depth 
        else:
            return sum([self.getSum(e, depth + 1) for e in elem.getList()])
#construct BST from preeorder traversal

def construct(preorder,start,end):
    if start > end: return None # pore vabchi
    node = TreeNode(preorder[start])
    i = start
    while i <= end:
        if preorder[i] > node.val:
            break
        i = i+1
    node.left = construct(preorder,start+1,i-1)
    node.right = construct(preorder,i,end)
    return node
        

class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        root = construct(preorder,0,len(preorder)-1)
        return root
#Given the root of a binary search tree, return a balanced binary search tree with the same node values. If there is more than one answer, return any of them.
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        
        def inorder(node,nodelist):
            if node:
                inorder(node.left,nodelist)
                nodelist.append(node.val)
                inorder(node.right,nodelist)
        
        def build(nodelist,start,end):
            if start > end: return None
            mid = (start+end)//2
            node = TreeNode(nodelist[mid])
            node.left = build(nodelist,start,mid-1)
            node.right = build(nodelist,mid+1,end)
            return node
        
        nodelist = []
        inorder(root,nodelist)
        root = build(nodelist,0,len(nodelist)-1)
        return root
#All Elements in Two Binary Search Trees
class Solution:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        values1,values2 = [],[]
        def inorder(node,nodelist):
            if node:
                inorder(node.left,nodelist)
                nodelist.append(node.val)
                inorder(node.right,nodelist)
                return nodelist
        nodelist = []
        
        list1 = inorder(root1,values1)
        list2 = inorder(root2, values2)
        i,j = 0,0
        result = []
        while i < len(values1) and j < len(values2):
            if list1[i] <= list2[j]:
                result.append(list1[i])
                i += 1
            else:
                result.append(list2[j])
                j += 1
        while i < len(values1):
            result.append(list1[i])
            i += 1
        while j < len(values2):
            result.append(list2[j])
            j += 1
        return result
#763 Partition labels 
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        dict = {}
        for i in range(len(s)):
            dict[s[i]] = i
        res = []
        start, last = 0,0
        for j in range(len(s)):
            last = max(last, dict[s[j]])
            if last == j:
                res.append(last-start+1)
                start = 1 + last
        return res 

#3 Lowest Common Ancestor of a Binary Tree III
class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node': 
        
        p_ancestors = []
        while p:
            p_ancestors.append(p)
            p = p.parent
        
        q_ancestors = []
        while q:
            q_ancestors.append(q)
            q = q.parent
            
        pParent = p_ancestors[::-1]
        qParent = q_ancestors[::-1]
        #lca = None
        for np,nq in zip(pParent,qParent):
            if np == nq: lca = np
        return lca
