import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_palette(sns.color_palette('tab20', 20))
# plt.style.use('dark_background')  # if background is dark theme
plt.rcParams['font.family'] = 'IPAexGothic'  # if using Japanese on the plot
plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['font.size'] = 12
%matplotlib inline
