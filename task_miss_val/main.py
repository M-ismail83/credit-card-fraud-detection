import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from miss_val import experimenter

test_results = experimenter()

dataframe_results = pd.DataFrame(test_results)

plt.figure(figsize=(14, 6))

# Define simple colors for the bars
bar_colors = ['gray', 'blue', 'orange', 'green', 'black']

plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
plt.bar(dataframe_results['Method'], dataframe_results['AUPRC'], color=bar_colors)
plt.title('AUPRC Scores')
plt.ylabel('Score')
plt.xticks(rotation=45) # Rotate names so they fit
plt.ylim(0, 1.1) # Set limit slightly above 1

plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
plt.bar(dataframe_results['Method'], dataframe_results['ROC-AUC'], color=bar_colors)
plt.title('ROC-AUC Scores')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.ylim(0, 1.1)

plt.tight_layout() # Fixes layout so labels don't get cut off
plt.savefig("task_miss_val/plots/imputation_results.png")
plt.show()