import numpy as np
from collections import defaultdict

class RNLJ:
    def __init__(self, T: np.ndarray, epsilon: float):
        self.T = np.asarray(T)  # ✅ Ensure T is a NumPy array and assign it to self.T
        self.epsilon = epsilon  
        self.n = len(T)

    def nljoin(self):
        tuples = []
        # Initialize LZC array
        lzc = np.zeros(self.n)
        lzc_zeros = np.ones(self.n)  # Start assuming all rows will have non-zero values
        # Use two dictionaries to store the current row and the row below it
        current_row_dict = defaultdict(int)
        next_row_dict = defaultdict(int)

        for i in range(self.n-1, -1, -1):  # Start from n-1 instead of n
            tuples = []
            for j in range(i-1, -1, -1):
                if self.T[i] - self.epsilon <= self.T[j] + self.epsilon and self.T[i] + self.epsilon >= self.T[j] - self.epsilon:
                    tuples.append((i, j))  # Correct tuple syntax

            if len(tuples) > 0:
                next_row_dict = current_row_dict
                current_row_dict = defaultdict(int)
                for (row,col) in tuples:
                    if row+1 <self.n and col+1 <self.n:
                        diagonal_value = next_row_dict.get(col + 1, 0)
                        current_row_dict[col] = min(diagonal_value + 1, row - col)

                        if current_row_dict[col] + row <= self.n - 1:
                            lzc[row] = max(current_row_dict[col] + 1, lzc[row])
                        else:
                            lzc_zeros[row] = 0  # Mark for zero in the final pass
                    else:
                        lzc_zeros[row] = 0  # Mark for zero in the final pass
                        current_row_dict[col] = 1
            else:
                lzc_zeros[i] = 0  # Mark for zero in the final pass
                #current_row_dict[col] = 1
                current_row_dict = defaultdict(int)
                next_row_dict = defaultdict(int)

        for i in range(self.n-1,-1,-1):
            if lzc_zeros[i] != 0:
                break
            lzc[i]=0

        return #lzc, lzc.sum()
