################
# Load modules #
################

import csv
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing

import os


'''
expected json format:
[
	{"fileName"     : "...",
	 "width"        : "...",
	 "height"       : "...",
	 "numOfCC"      : "...",
	 "numOfZone"    : "...",
	 "zoneQ1"       : "...",
	 "zoneQ3"       : "...",
	 "zoneMax"      : "...",
	 "zoneMin"      : "...",
	 "zoneIRQ"      : "...",
	 "prediction"   : "...",
	 "contrast"     : "...",
	 "rangeeffect"  : "...",
	 "bleedthrough" : "...",
	},
	...
]
'''


# Utils
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .6), xycoords=ax.transAxes,
               size = 24)


json_file_dhsegment = 'PATH/TO/METADATA/JSON'
#json_file_out = 'output_17k_voronoi_dhSegment_zone_diqa.json'

# Read JSON
with open(json_file_out) as json_file:
    data = json.load(json_file)
print("Total {} items.\n".format(len(data)))

# JSON to Dataframe
df = pd.DataFrame.from_records(data)
df.info()

# Drop fileName
df_drop = df.drop(columns=['fileName'])
df_drop = df_drop.convert_objects(convert_numeric=True)

# Add new features
my_df = df_drop.copy()
my_df['resolution'] = df_drop['width']*df_drop['height']

my_df['zoneMinBound'] = df_drop['zoneQ1']-1.5*df_drop['zoneIQR']
my_df['zoneMaxBound'] = df_drop['zoneQ3']+1.5*df_drop['zoneIQR']
my_df['*zoneOutlierity'] = ((df_drop['zoneMax']-my_df['zoneMaxBound'])/(my_df['zoneMax']-my_df['zoneMin']))

my_df['*density'] = my_df['numOfSites']/my_df['resolution']
my_df['**density'] = my_df['numOfSites']*my_df['sizeMean']/my_df['resolution']
my_df['***density'] = my_df['numOfZone']/my_df['resolution']
my_df['****density'] = my_df['numOfNoiseZone']/my_df['resolution']

my_df['*bleedthrough'] = my_df['bleedthrough']/my_df['resolution']
my_df['*rangeeffect'] = my_df['rangeeffect']/my_df['resolution']

my_df['*contrast'] = my_df['contrast']/my_df['resolution']

my_df.info()


# Analyze Correlationship
most_correlated = my_df.corr().abs()['difficulty'].sort_values(ascending=False)
print("Most Correlated Features:\n")
print(most_correlated)


# Further explore on the most correlated featurse
if(False):
	_my_df = my_df[['difficulty','*density','contrast','numOfCC','numOfZone','*zoneOutlierity','bleedthrough','rangeeffect','prediction']]
	_my_df = _my_df.rename(columns={"*density": "density*", "numOfCC": "number of letters", "numOfZone": "number of zones*", "*zoneOutlierity":"zone size abnormality*","bleedthrough":"bleed-through","rangeeffect":"range-effect","prediction":"prediction*"})
	_my_df.hist(bins=50, figsize=(20,15))

if(False)
	cmap = sns.cubehelix_palette(light=1, dark = 0.1,
                             hue = 0.5, as_cmap=True)

	sns.set_context(font_scale=2)

	# Pair grid set up
	g = sns.PairGrid(_my_df)

	# Scatter plot on the upper triangle
	g.map_upper(plt.scatter, s=10, color = 'red')

	# Distribution on the diagonal
	g.map_diag(sns.distplot, kde=False, color = 'red')

	# Density Plot and Correlation coefficients on the lower triangle
	g.map_lower(sns.kdeplot, cmap = cmap)
	g.map_lower(corrfunc);
