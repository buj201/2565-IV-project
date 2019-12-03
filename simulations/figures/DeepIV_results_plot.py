import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

sns.set_context('notebook', font_scale=1.5)

results = glob.glob('output/MSEs/DeepIV_no_MINST_simulation/*.csv')

results = pd.concat([
    pd.read_csv(result) for result in results
],axis=0)

estimator_names = {
    'DeepIV_MSE' : 'DeepIV',
    'forest_MSE' : 'Instrumental Forest',
    'iv_series_MSE' : 'Series estimator'
}
results.rename(columns=estimator_names,inplace=True)

new_cols = list(estimator_names.values())

melted = pd.melt(frame=results, id_vars=['n','ypcor'], value_vars=new_cols)

melted.rename(columns={'ypcor':'$\\rho_{y,p}$',
                       'n':'$n$',
                       'value':"$\\hat{E}\\left[\\left(\\hat{\\tau}(X) - \\tau(X)\\right)^2\\right]$",
                       'variable':"Estimator"},inplace=True)

g = sns.catplot(x='$n$', y='$\\hat{E}\\left[\\left(\\hat{\\tau}(X) - \\tau(X)\\right)^2\\right]$',
                hue='Estimator', col='$\\rho_{y,p}$', data=melted, kind='box',
                showfliers=False, dodge=True, height=4, aspect=1, legend=False)

plt.legend(bbox_to_anchor=(0.53,-0.16), loc="lower center", title='Estimator', 
           bbox_transform=g.fig.transFigure, ncol=3)
g.fig.suptitle('Causal MSE by $n$ and $\\rho_{y,p}$',y=1.03)
g.savefig('output/figures/DeepIV_results.png')