import pickle
import numpy as np
from utils import plot_learning_curve_with_shadow

# Load scores
with open('lunarlander_online_scores.pkl', 'rb') as f:
    scores = pickle.load(f)

# Generate x-axis
x = [i + 1 for i in range(len(scores))]

# Plot and save
plot_learning_curve_with_shadow(x, scores, 'plots/lunarlander_online_shadow_plot.png')
print("✅ Shadow plot saved as 'plots/lunarlander_online_shadow_plot.png'")
