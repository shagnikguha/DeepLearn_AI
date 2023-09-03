"""
INDEXING: means referring to an element of an array by its position within the array.
"""

import numpy as np

#vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

#access an element
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

#indexs must be within the range of the vector or they will produce and error
try:
    c = a[10]  
except Exception as e:
    print("The error message you'll see is:")
    print(e)