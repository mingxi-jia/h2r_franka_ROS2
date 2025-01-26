import numpy as np
import pandas as pd

# Define the objects
objects = [# kitchen
           "strawberry", 
           "banana", 
           "lemon", 
           "spam meat can",
           "tomato soup can",
           "red mug's handle", 
           "red mug's rim",
           "gray mug's handle", 
           "gray mug's rim",
           # others
           "green shoe's toe",
           "green shoe's collar",
           "orange block",
           "blue block",
           # office
           "green paint bottle's cap",
           "green paint bottle's body",
           "pink paint bottle's cap",
           "pink paint bottle's body",
           "black tape",
           "brown tape",
           ]
num_objects = len(objects)

# Initialize a 3x3 matrix with zeros
M = np.zeros((num_objects, num_objects))

# Set the known similarity values
M[5, 7] = M[7, 5] = 0.9  # similarity between mug's handles
M[6, 8] = M[8, 6] = 0.9  # similarity between mug's rims
M[11, 12] = M[12, 11] = 0.9  # similarity between blocks
M[13, 15] = M[15, 13] = 0.9  # paint bottle's cap
M[14, 16] = M[16, 14] = 0.9  # paint bottle's body
M[15, 16] = M[16, 15] = 0.9 # tape

# Set the diagonal elements to 1 (since the similarity of an object with itself is always 1)
np.fill_diagonal(M, 1)

# Convert the matrix into a DataFrame for better readability
M_df = pd.DataFrame(M, index=objects, columns=objects)

# Display the matrix
print("Similarity Matrix:")
print(M_df)

similarity = M_df.loc["banana", "lemon"]
print(f"query result is {similarity}")

similarity = M_df.loc["red mug's handle", "gray mug's handle"]
print(f"query result is {similarity}")