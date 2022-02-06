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
