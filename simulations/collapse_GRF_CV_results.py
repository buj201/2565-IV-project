import pandas as pd
import glob

def parse_path(p):
    parsed = [pair.split('_') for pair in p.split('.csv')[0].split('__')]
    return pd.Series({k:v for k,v in parsed})

def get_stage_results(stage):
    stage_results = pd.concat([
        pd.read_csv(f).assign(path=f.split(f'STAGE{stage}__')[1]) \
        for f in glob.glob(f'output/MSEs/GRF_simulation_CV/CVregularizers_True__STAGE{stage}__*.csv')
    ], axis=0)
    stage_results = pd.concat([stage_results.drop('path',axis=1),
                               stage_results.path.apply(parse_path)],axis=1)
    return stage_results

if __name__=='__main__':
    get_stage_results(1).to_csv('output/MSEs/GRF_simulation_CV/STAGE1_all_results.csv',index=False)
    get_stage_results(2).to_csv('output/MSEs/GRF_simulation_CV/STAGE2_all_results.csv',index=False)