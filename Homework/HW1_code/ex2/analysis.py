
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# avg_monthly_precip = pd.read_csv('avg-precip-months-seasons.csv', sep=";")
# READ FROM CSV TO PANDAS FRAME
output_frame = pd.read_csv('output_frame2.csv', sep=";")
print(output_frame)


# BUILD TABLE ANOVA
model = ols('time ~ C(core) + C(estimator) + C(core):C(estimator)', data=output_frame).fit()
print(sm.stats.anova_lm(model, typ=2))