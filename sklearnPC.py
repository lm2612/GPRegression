import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ImportData import *
from PCA_function import *
from sklearn.decomposition import PCA
pca = PCA()
fitpca = pca.fit(X_SfcTemp)

var_exp = fitpca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)
plt.figure(figsize=(6, 4))
plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
plt.step(range(len(var_exp)), cum_var_exp, where='mid',
             label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('/home/lm2612/GPRegression/pc_varexp_bar_line.png')

