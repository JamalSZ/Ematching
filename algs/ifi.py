
from sortedcontainers import SortedDict  # Efficient sorted dictionary
import numpy as np
from collections import defaultdict


class IFI:
    def __init__(self, T: np.ndarray, epsilon: float):
        self.T = np.asarray(T)  # ✅ Ensure T is a NumPy array and assign it to self.T
        self.epsilon = epsilon  
        self.n = len(T)

    def bisect_left(self,arr, x, lo=0, hi=None):
        """
        Locate the insertion point for x in a sorted sequence arr.
        The parameters lo and hi may be used to specify a subset of the sequence
        to search.
        """
        if hi is None:
            hi = len(arr)

        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < x:
                lo = mid + 1
            else:
                hi = mid

        return lo

    def bisect_right(self, arr, x, lo=0, hi=None):
        """
        Return the index after the last occurrence of x in a sorted sequence arr.
        The parameters lo and hi may be used to specify a subset of the sequence
        to search.
        """
        if hi is None:
            hi = len(arr)

        while lo < hi:
            mid = (lo + hi) // 2
            if x < arr[mid]:
                hi = mid
            else:
                lo = mid + 1

        return lo

    def LZ2lookup_ifi(self,keys, D, s, j, e):
        """Find indices i < j where T[j] is within e-range of T[i]."""
        matches = set()
        
        left, right = self.bisect_left(keys, s - e), self.bisect_right(keys, s + e)
        
        for key in keys[left:right]:
            values = D.get(key, [])
            for k in values:
                if k < j:
                    matches.add(k)
                else:
                    break

        return matches

    def LZ2build_IIdx_ifi(self):
        """Build an inverted index for quick lookup."""
        index = {}
        for i, char in enumerate(self.T):
            index.setdefault(char, []).append(i)
        
        return SortedDict(index)  # Keeps keys sorted for fast range queries

    def LZ2_ifi(self):
        """Compute pairs (cur, m) where T[cur] is within tolerance e of T[m]."""
        Idx = self.LZ2build_IIdx_ifi()
        keys = list(Idx.keys())
        result = []
        ee = 2 * self.epsilon
        
        lzc = np.zeros(self.n, dtype=int)
        lzc_zeros = np.ones(self.n, dtype=int)  # Start assuming all rows will have non-zero values
        # Use two dictionaries to store the current row and the row below it
        current_row_dict = defaultdict(int)
        next_row_dict = defaultdict(int)

        for cur in range(self.n-1,-1,-1):
            row = cur
            matches = self.LZ2lookup_ifi(keys, Idx, self.T[cur], cur, ee)
            
            if len(matches) > 0:
                next_row_dict = current_row_dict
                current_row_dict = defaultdict(int)
                for col in matches:
                    if row+1 <self.n and col+1 <self.n:
                        diagonal_value = next_row_dict.get(col + 1, 0)
                        current_row_dict[col] = min(diagonal_value + 1, row - col)

                        if current_row_dict[col] + row <= self.n - 1:
                            lzc[row] = max(current_row_dict[col] + 1, lzc[row])
                        else:
                            #print(row,col, diagonal_value,current_row_dict[col])
                            lzc_zeros[row] = 0  # Mark for zero in the final pass
                    else:
                        lzc_zeros[row] = 0  # Mark for zero in the final pass
                        current_row_dict[col] = 1
            else:
                lzc_zeros[row] = 0  # Mark for zero in the final pass
                current_row_dict = defaultdict(int) 
                next_row_dict = defaultdict(int)
        for i in range(self.n-1,-1,-1):
            if lzc_zeros[i] != 0:
                break
            lzc[i]=0

        return lzc, lzc.sum()
