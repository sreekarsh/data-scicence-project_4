# project
# Set up Seaborn style for consistency in plots
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.boxplot(y=boston_df['MEDV'])
plt.title("Boxplot of Median Value of Owner-Occupied Homes")
plt.ylabel("Median Value (in $1000's)")
plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(x='CHAS', data=boston_df)
plt.title("Count of Homes Bounded by the Charles River")
plt.xlabel("Charles River Proximity (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Discretize AGE into groups
boston_df['AGE_group'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, 100], labels=['<=35', '35-70', '>70'])

plt.figure(figsize=(8, 6))
sns.boxplot(x='AGE_group', y='MEDV', data=boston_df)
plt.title("Boxplot of MEDV by AGE Group")
plt.xlabel("Age Group")
plt.ylabel("Median Value (in $1000's)")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='INDUS', y='NOX', data=boston_df)
plt.title("Scatter Plot of NOX vs. INDUS")
plt.xlabel("Proportion of Non-Retail Business Acres (INDUS)")
plt.ylabel("Nitric Oxide Concentration (NOX)")
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(boston_df['PTRATIO'], kde=True)
plt.title("Histogram of Pupil-Teacher Ratio (PTRATIO)")
plt.xlabel("Pupil-Teacher Ratio")
plt.ylabel("Frequency")
plt.show()

# Group data based on CHAS
medv_chas1 = boston_df[boston_df['CHAS'] == 1]['MEDV']
medv_chas0 = boston_df[boston_df['CHAS'] == 0]['MEDV']

# Perform t-test
t_stat, p_value_ttest = stats.ttest_ind(medv_chas1, medv_chas0)
print("T-Test Results:")
print("t-statistic:", t_stat)
print("p-value:", p_value_ttest)

# One-Way ANOVA Test for MEDV across AGE groups
age_groups = [boston_df[boston_df['AGE_group'] == grp]['MEDV'] for grp in boston_df['AGE_group'].cat.categories]
f_stat, p_value_anova = stats.f_oneway(*age_groups)
print("ANOVA Results:")
print("F-statistic:", f_stat)
print("p-value:", p_value_anova)

correlation_coef, p_value_corr = stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])
print("Pearson Correlation Results:")
print("Correlation Coefficient:", correlation_coef)
print("p-value:", p_value_corr)

# Simple Linear Regression
regression_model = ols('MEDV ~ DIS', data=boston_df).fit()

# Print out regression summary
print(regression_model.summary())
