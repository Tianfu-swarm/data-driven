import numpy as np
import matplotlib.pyplot as plt

# Define the function to compute k_i based on the recursive formula
def compute_k(i, n_follower, k_values):
    if i < 3:
        return None  # Formula starts from i >= 3
    sum_k_before = np.sum(k_values[:i-2])
    sum_k_before_2 = np.sum(k_values[:i-3]) if i > 3 else 0
    k_prev = k_values[i-2] if i > 2 else 0

    # Avoid division by zero
    if n_follower - sum_k_before_2 == 0:
        return np.nan  # Returning NaN to avoid invalid division

    k_i = (n_follower - sum_k_before) * (k_prev / (n_follower - sum_k_before_2))
    return k_i

# Parameters
n_follower = 86  # Total number of followers
num_steps = 30  # Number of iterations (steps) to calculate k_i

# Initialize k_values with k1, k2, and an empty list for further steps
k_1 = 20  # k_1 = 24 (starting condition)
k_2 = n_follower * k_1 / (n_follower + k_1)
k_values = []
k_values.append(k_1)
k_values.append(k_2)

k_all = []
k_all.append(k_1)
k_all_now = k_1+k_2
k_all.append(k_all_now)

# Calculate k_3, k_4, ..., k_n using the recursive formula
for i in range(3, num_steps + 1):
    k_i = compute_k(i, n_follower, k_values)
    k_all_now = k_all_now + k_i
    k_values.append(k_i)
    k_all.append(k_all_now)

# Plotting the results to visualize convergence
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_steps + 1), k_values, marker='o', linestyle='-', color='b')
plt.xticks(np.arange(0, num_steps + 1, 1))
plt.yticks(np.arange(0, k_values[0]+2, 2))
plt.title(r"Convergence of $k_i$ values")  # Use raw string for LaTeX
plt.xlabel("layer (i)")
plt.ylabel("$k_i$")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(range(1, num_steps + 1), k_all, marker='o', linestyle='-', color='r')
plt.xticks(np.arange(0, num_steps + 1, 1))
plt.yticks(np.arange(k_1, k_all_now + 1, 5))
plt.title("Sum of all particles will join to group with leader")  # Use raw string for LaTeX
plt.xlabel("layer (i)")
plt.ylabel("sum of all particles")
plt.grid(True)
plt.show()