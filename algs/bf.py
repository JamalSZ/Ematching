
import numpy as np

class BF:
    def __init__(self, T: np.ndarray, epsilon: float):
        self.T = np.asarray(T)  # ✅ Ensure T is a NumPy array and assign it to self.T
        self.epsilon = epsilon  
        self.n = len(T)

    def overlap(x0,x1,tol):
        st1,ed1 = x0-tol, x0+tol
        st2,ed2 = x1-tol, x1+tol
        if st2<=ed1 and st1 <= ed2:
            return True
        else: 
            return False

    def lzcnl_numeric(self):
        output = np.zeros(self.n)
        output[0] = 1
        
        for start_idx in range(1,self.n):
            max_subsequence_matched = 0
            for i in range(0,start_idx):
                j = 0
                end_distance = self.n - start_idx 
                
                while( (start_idx+j < self.n) and (i+j < start_idx) and self.overlap(self[i+j],self.T[start_idx+j],self.epsilon)>0 ):
                    j = j + 1
                if j == end_distance: 
                    output[start_idx:] = 0
                    return #output,sum(output)
                if j > max_subsequence_matched:
                    max_subsequence_matched = j
            output[start_idx] = max_subsequence_matched + 1

        return #output,sum(output)