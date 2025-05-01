import numpy as np
from bisect import bisect_left, bisect_right
from typing import List, Tuple, Dict
from collections import defaultdict

class IESJ:
    def __init__(self, T: np.ndarray, epsilon: float):
        self.T = np.asarray(T)  # ✅ Ensure T is a NumPy array and assign it to self.T
        self.epsilon = epsilon
        self.n = len(T)
    def revlzc(self, tuples_set: list) -> Tuple[np.ndarray, float]:
        n = self.n
        if not tuples_set:
            return np.ones(n, dtype=float), float(n)

        # Initialize LZC array
        lzc = np.zeros(n, dtype=float)
        lzc_zeros = np.ones(n, dtype=int)  # Start assuming all rows will have non-zero values

        # Initialize variables for processing
        first_row = tuples_set[0][0]
        lzc[:first_row] = 1  # Mark rows before the first row in tuples_set as 1

        # Use two dictionaries to store the current row and the row below it
        current_row_dict = defaultdict(int)
        next_row_dict = defaultdict(int)

        previous_row = n-1

        # Process tuples in reverse order
        for row, col in reversed(tuples_set):

            if previous_row-row > 0:
                next_row_dict = current_row_dict
                current_row_dict = defaultdict(int)
            
            if row < previous_row - 1:
                next_row_dict = defaultdict(int)#current_row_dict
                current_row_dict = defaultdict(int)
                lzc[row + 1:previous_row] = 1  # Mark rows between as 1

            if row + 1 < n and col + 1 < n:
                # Get value from the next row dictionary, defaulting to 0 if not present
                diagonal_value = next_row_dict.get(col + 1, 0)
                current_row_dict[col] = min(diagonal_value + 1, row - col)
                if current_row_dict[col] + row <= n - 1:
                    lzc[row] = max(current_row_dict[col] + 1, lzc[row])
                else:
                    lzc_zeros[row] = 0  # Mark for zero in the final pass
            else:
                lzc_zeros[row] = 0  # Mark for zero in the final pass
                current_row_dict[col] = 1

            # Update the next row dictionary for the next iteration
            

            previous_row = row

        # Update LZC based on the zeros flag
        
        for i in range(n-1,-1,-1):
            if lzc_zeros[i] != 0:
                break
            lzc[i]=0

        

        return #lzc, lzc.sum()


    def get_permutation_array(L1_indices: np.ndarray, L2_indices: np.ndarray) -> List[int]:
        """Return corresponding positions of elements of L1 in L2 based on their positional indexes.
        
        Args:
            L1_indices: Array of indices for list L1
            L2_indices: Array of indices for list L2
            
        Returns:
            Permutation array mapping L2 indices to their positions in L1
        """
        index_map = {val: i for i, val in enumerate(L1_indices)}
        return [index_map[val] for val in L2_indices]


    def get_offset_array_op1(L1: np.ndarray, L1_prime: np.ndarray, op1: str) -> List[int]:
        """Return an offset array by comparing L1 and L1' based on op1.
        
        Args:
            L1: First array of values
            L1_prime: Second array of values to compare against
            op1: Comparison operator ('lt', 'le', 'gt', 'ge')
            
        Returns:
            Offset array containing indices where the comparison holds
        """
        operators = {
            'lt': (lambda x, y: x < y),
            'le': (lambda x, y: x <= y),
            'gt': (lambda x, y: x > y),
            'ge': (lambda x, y: x >= y)
        }
        
        if op1 not in operators:
            raise ValueError(f"Invalid operator '{op1}'. Must be one of {list(operators.keys())}")
        
        return [bisect_left(L1_prime, val) for val in L1]


    def get_offset_array_op2(L2: np.ndarray, L2_prime: np.ndarray, op2: str) -> List[int]:
        """Return an offset array by comparing L2 and L2' based on op2.
        
        Args:
            L2: First array of values
            L2_prime: Second array of values to compare against
            op2: Comparison operator ('gt' or 'ge')
            
        Returns:
            Offset array containing indices where the comparison holds
        """
        if op2 not in ['gt', 'ge']:
            raise ValueError("Operator must be either 'gt' or 'ge'")
        
        if op2 == 'gt':
            return [bisect_left(L2_prime, val) - 1 for val in L2]
        else:  # 'ge'
            return [bisect_right(L2_prime, val) - 1 for val in L2]


    def check_condition(L3: np.ndarray, L3_prime: np.ndarray, i: int, k: int, op: str) -> bool:
        """Check if the condition L3[i] op L3_prime[k] holds.
        
        Args:
            L3: First array of values
            L3_prime: Second array of values
            i: Index in L3
            k: Index in L3_prime
            op: Comparison operator ('lt', 'le', 'gt', 'ge')
            
        Returns:
            Boolean result of the comparison
        """
        operators = {
            'gt': (lambda x, y: x > y),
            'ge': (lambda x, y: x >= y),
            'lt': (lambda x, y: x < y),
            'le': (lambda x, y: x <= y)
        }
        
        if op not in operators:
            raise ValueError(f"Invalid operator '{op}'. Must be one of {list(operators.keys())}")
        
        return operators[op](L3[i], L3_prime[k])


    def create_dataframe(self) -> Dict[str, np.ndarray]:
        """Create a dictionary representing a dataframe with time, start and end values.
        
        Args:
            T: Input time series data
            tol: Tolerance value for creating intervals
            
        Returns:
            Dictionary with 't', 's', and 'e' arrays
        """
        return {
            't': np.arange(self.n),
            's': self.T - self.epsilon,
            'e': self.T + self.epsilon
        }


    def iejoin(self) -> List[Tuple[int, int]]:
        """Inequality join with two inequality predicates.
        
        Args:
            T: Input time series data
            epsilon: Tolerance value for the join conditions
            
        Returns:
            List of tuple pairs that satisfy the join conditions
        """
        T = self.create_dataframe()
        # Initialize arrays
        L1 = T['s']
        L2 = T['t']
        L3 = T['e']
        
        L1_prime, L2_prime, L3_prime = L3, L2, L1
        
        # Get sorted indices
        L1_indices = np.argsort(L1)  # ASC order
        L1_prime_indices = np.argsort(L1_prime)
        L2_indices = np.argsort(L2)  # ASC order

        # Compute permutation and offset arrays
        P = self.get_permutation_array(L1_indices, L2_indices)
        O1 = self.get_offset_array_op1(L1[L1_indices], L1_prime[L1_prime_indices], 'le')
        
        O2 = range(-1, self.n-1)  # Equivalent to the commented get_offset_array_op2 call
        
        # Initialize bit array and result storage
        B_prime = np.zeros(self.n, dtype=bool)
        join_result = []
        
        # Main processing loop
        for i in range(1, self.n):
            # Set bits for tuples in T' with Y' < Y of current tuple in T
            off2 = O2[i]
            B_prime[P[off2]] = True
            
            # Check other join condition
            off1 = O1[P[i]]
            for k in range(off1, self.n):
                left = L2[L2_indices[i]]
                right = L2[L1_prime_indices[k]]
                
                if B_prime[k] and self.check_condition(L3, L3_prime, left, right, 'ge'):
                    join_result.append((left, right))
                
                # Early termination if we've passed possible matches
                if L1_prime[right] > L1[left] + 4 * self.epsilon:
                    break

        join_results = sorted(join_result)
        _ = self.revlzc(join_results)
        
        return 
