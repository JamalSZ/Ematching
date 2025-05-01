import numpy as np
from collections import defaultdict

class AVLNode:
    def __init__(self, key, entries=None):
        self.key = key  # First element of the segment
        self.entries = entries or []  # list of (start_idx, length)
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def __init__(self):
        self.root = None

    def insert(self, root, key, entry):
        if not root:
            return AVLNode(key, [entry])
        if key < root.key:
            root.left = self.insert(root.left, key, entry)
        elif key > root.key:
            root.right = self.insert(root.right, key, entry)
        else:
            root.entries.append(entry)
            return root

        root.height = 1 + max(self.height(root.left), self.height(root.right))
        balance = self.balance(root)

        if balance > 1 and key < root.left.key:
            return self.rotate_right(root)
        if balance < -1 and key > root.right.key:
            return self.rotate_left(root)
        if balance > 1 and key > root.left.key:
            root.left = self.rotate_left(root.left)
            return self.rotate_right(root)
        if balance < -1 and key < root.right.key:
            root.right = self.rotate_right(root.right)
            return self.rotate_left(root)

        return root

    def insert_entry(self, key, entry):
        self.root = self.insert(self.root, key, entry)

    def query(self, root, low, high):
        if not root:
            return []
        results = []
        if low <= root.key <= high:
            results.extend(root.entries)
        if low < root.key:
            results += self.query(root.left, low, high)
        if high > root.key:
            results += self.query(root.right, low, high)
        return results

    def search_range(self, low, high):
        return self.query(self.root, low, high)

    def height(self, node):
        return node.height if node else 0

    def balance(self, node):
        return self.height(node.left) - self.height(node.right)

    def rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.height(z.left), self.height(z.right))
        y.height = 1 + max(self.height(y.left), self.height(y.right))
        return y

    def rotate_right(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self.height(z.left), self.height(z.right))
        y.height = 1 + max(self.height(y.left), self.height(y.right))
        return y

class FIT:
    def __init__(self, series, epsilon):
        self.epsilon = epsilon
        self.series = series  # Store reference to the original series
        self.length = len(series)
        self.forest = defaultdict(AVLTree)  # Maps segment lengths to AVL trees

    def overlap(self, a, b):
        return a - self.epsilon <= b + self.epsilon and a + self.epsilon >= b - self.epsilon

    def match(self, t):
        max_match = 0
        n = self.length

        for l in sorted(self.forest.keys()):
            if t + l > n:
                continue  # Skip segments that would exceed series length

            # Get the first element of the query segment
            x0 = self.series[t]
            
            # Find candidate segments with similar first elements
            candidates = self.forest[l].search_range(x0 - 2*self.epsilon, x0 + 2*self.epsilon)

            for cand_start, cand_length in candidates:
                # Compare all elements in the segments
                match_found = True
                for i in range(l):
                    if not self.overlap(self.series[cand_start + i], self.series[t + i]):
                        match_found = False
                        break
                
                if match_found:
                    max_match = max(max_match, l)
                    break  # Found the longest possible match for this length

        return max_match

    def insert_all_prefixes(self, start, t):
        for i in range(start, t + 1):
            seg_length = t - i + 1
            x0 = self.series[i]  # First element of the segment
            # Store just the start index and length instead of the full segment
            self.forest[seg_length].insert_entry(x0, (i, seg_length))

    def run_fit(self):
        n = self.length
        lambda_t = np.zeros(n)
        lambda_t[0] = 1
        self.insert_all_prefixes(0, 0)

        for t in range(1, n):
            match_len = self.match(t)
            self.insert_all_prefixes(0, t)
            
            if match_len + t >= n:
                lambda_t[t:] = 0
                break
            else:
                lambda_t[t] = match_len + 1

        return #lambda_t, np.sum(lambda_t)