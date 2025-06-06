import numpy as np
import matplotlib.pyplot as plt
import os

def plot_learning_curve_with_shadow(x, scores, figure_file):
    scores = np.array(scores)
    running_avg = np.zeros(len(scores))
    std_dev = np.zeros(len(scores))
    
    for i in range(len(scores)):
        window = scores[max(0, i - 100):(i + 1)]
        running_avg[i] = np.mean(window)
        std_dev[i] = np.std(window)
    
    plt.figure()
    plt.plot(x, running_avg, label='Running Avg (100)')
    plt.fill_between(x, running_avg - std_dev, running_avg + std_dev, alpha=0.2, label='±1 Std Dev')
    plt.title('Learning Curve with Running Average and Shadow')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()

    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    plt.savefig(figure_file)
    plt.close()
