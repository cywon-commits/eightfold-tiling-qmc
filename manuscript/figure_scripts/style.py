import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':8,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
    'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix','axes.linewidth':0.7,
    'xtick.major.width':0.6,'ytick.major.width':0.6,'lines.linewidth':1.3,'savefig.bbox':'tight','savefig.pad_inches':0.02})
COL=3.40; DCOL=7.05
ZC={3:'#f4a261',4:'#ffffff',5:'#8ecae6',6:'#2a7ab8',7:'#1d3557',8:'#000000'}
TILE={'sq':'#f3e3b6','rh':'#d9e8f5'}
def label(ax,s,x=-0.02,y=1.02): ax.text(x,y,s,transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom',ha='left')
