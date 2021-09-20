class Solution:
    # @param A : integer
    # @param B : integer
    # @param C : integer
    # @param D : integer
    # @param E : integer
    # @param F : integer
    # @return an integer
    def knight(self, A, B, C, D, E, F):
        from collections import deque
        seen = set()
        q = deque([(C, D)])
        res = 0
        while q:
            for _ in range(len(q)):
                i, j = q.popleft()
                if i == E and j == F:
                    return res
                for di, dj in (2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2):
                    if 1 <= i + di <= A and 1 <= j + dj <= B and (i+di, j+dj) not in seen:
                        seen.add((i+di, j+dj))
                        q.append((i+di, j+dj))
            res += 1
        return -1
