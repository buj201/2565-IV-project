import pandas as pd
import numpy as np
import glob

results = glob.glob('output/MSEs/GRF_simulation/*.csv')

results = pd.concat([
    pd.read_csv(result) for result in results
],axis=0)

groupers = ['kappa','p','n','omega','additive']

grouper_names = {'kappa':'$\kappa$','omega':'$\omega$','additive':'Additive'}

groupers = [x if (x not in grouper_names.keys()) else grouper_names[x]\
            for x in groupers]

mses = {'iv_series_MSE': 'IV series',
        'forest_MSE': 'GRF',
        'DeepIV_MSE': 'DeepIV'}

results.rename(columns=mses,inplace=True)
results.rename(columns=grouper_names,inplace=True)

atts = list(mses.values())

sizes = results.groupby(groupers).size()
means = results.groupby(groupers)[atts].mean().round(3)
ses = results.groupby(groupers)[atts].std() / pd.concat([
    np.sqrt(sizes).rename(x) for x in atts
],axis=1)

def order(stats):
    ordered = stats.unstack('Additive')\
                   .reorder_levels([1,0],axis=1)\
                   .reindex([True,False],level='Additive',axis=1)
    
    return ordered

def get_min_in_row(stats,additive):
    s = stats.loc[:,pd.IndexSlice[additive,:]]
    return s.idxmin(axis=1)
def add_bolds(full_stats):
    for ix, col in get_min_in_row(order(means),additive=True).iteritems():
        full_stats.loc[ix, col] = '\\textbf{' + full_stats.loc[ix, col] + '}'
        
    for ix, col in get_min_in_row(order(means),additive=False).iteritems():
        full_stats.loc[ix, col] = '\\textbf{' + full_stats.loc[ix, col] + '}'
        
    return full_stats


final_table = order(means).astype(str) + ' (' + order(ses.round(3)).astype(str) + ')'
final_table = add_bolds(final_table)
final_table.to_latex('output/tables/GRF_results.tex', escape=False)