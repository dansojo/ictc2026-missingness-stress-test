"""Descriptive participant-effect figures from verified full summary tables."""
from pathlib import Path
import argparse
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
METHODS=['CD_ALL_DAY','CD_TIME_MEAN','CD_TIME_MEDIAN','CD_WEEKTYPE_MEAN','CD_WEEKTYPE_MEDIAN','CD_PAST_TIME_MEAN','CD_PAST_RECENT_MEAN','CD_SIMILAR_COPY']
LABELS=['Other-day mean','Same-time mean','Same-time median','Weektype mean','Weektype median','Past-time mean','Recent-past mean','Similar-day copy']
PRIMITIVES=['screen_load_24h','phone_activity_load_24h','usage_load_24h','mobile_light_exposure_24h','wearable_light_exposure_24h']
TITLES=['Screen state','Activity state','App usage','Phone light','Wearable light']

def run(summary,output):
    output.mkdir(exist_ok=False,parents=True)
    frame=pd.read_csv(summary/'summary.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for comparator,label in [('M0','No repair'),('CD_ALL_DAY','Other-day whole-day mean')]:
        fig,axes=plt.subplots(1,5,figsize=(18,6),sharey=True,layout='constrained')
        for ax,primitive,title in zip(axes,PRIMITIVES,TITLES):
            ax.axvline(0,color='#777777',lw=.8,zorder=0)
            for geometry,dy,color,name in [('contiguous_20pct',-.12,'#a64c28','Contiguous'),('scattered_random_20pct',.12,'#236b91','Scattered')]:
                rows=frame.loc[frame.primitive.eq(primitive)&frame.comparator.eq(comparator)&frame.geometry.eq(geometry)].set_index('method').reindex(METHODS)
                for y,r in enumerate(rows.itertuples()):
                    if not np.isfinite(r.median_gain):
                        if geometry=='contiguous_20pct':ax.text(.98,y,'N/A',transform=ax.get_yaxis_transform(),ha='right',va='center',color='#777777',fontsize=8)
                        continue
                    ax.plot([r.ci_low,r.ci_high],[y+dy,y+dy],color=color,lw=1.4)
                    ax.scatter(r.median_gain,y+dy,s=24,color=color,label=name if y==0 else None,zorder=3)
            ax.set_title(title,fontweight='bold');ax.grid(axis='y',alpha=.12);ax.set_xlabel('Standardized error reduction')
            ax.tick_params(axis='x',labelrotation=30)
        axes[0].set_yticks(range(len(METHODS)),LABELS);axes[0].invert_yaxis()
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.055),ncol=2,frameon=False)
        fig.suptitle(f'Cross-day reconstruction versus {label}\nMedian of 10 participant medians; 95% participant-bootstrap intervals',fontsize=14)
        fig.savefig(output/f'gain_vs_{comparator}.png',dpi=160,bbox_inches='tight')
        fig.savefig(output/f'gain_vs_{comparator}.svg',bbox_inches='tight')
        plt.close(fig)
    print(str(output))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,default=HERE/'summary_01');p.add_argument('--output',type=Path,default=HERE/'figures_02')
    a=p.parse_args();run(a.summary,a.output)
