import matplotlib.pyplot as plt

# Environment transitions used
online_env_steps = 1_280_000  # Online SAC directly interacts
offline_env_steps = 1_280_000  # Collected once, reused many times

# Final scores
online_score = 275
offline_score = 200

# True sample efficiency = final score / environment interactions used
online_eff = online_score / (online_env_steps / 1000)
offline_eff = offline_score / (offline_env_steps / 1000)

# Plot
methods = ['Online SAC', 'Offline SAC']
efficiencies = [online_eff, offline_eff]

plt.figure(figsize=(6, 4))
plt.bar(methods, efficiencies, color=['blue', 'green'])
plt.ylabel('Score per 1,000 Environment Steps')
plt.title('True Sample Efficiency (Env Steps Only)')
plt.grid(axis='y')
plt.savefig('plots/sample_efficiency_env_based.png')
plt.show()
