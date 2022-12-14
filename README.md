# Description

Whether out at a restaurant or buying tickets to a concert, modern life counts on the convenience of a credit card to make daily purchases.
It saves us from carrying large amounts of cash and also can advance a full purchase that can be paid over time.
How do card issuers know we’ll pay back what we charge?
That’s a complex problem with many existing solutions—and even more potential improvements, to be explored in this competition.

Credit default prediction is central to managing risk in a consumer lending business.
Credit default prediction allows lenders to optimize lending decisions, which leads to a better customer experience and sound business economics.
Current models exist to help manage risk. But it's possible to create better models that can outperform those currently in use.

American Express is a globally integrated payments company.
The largest payment card issuer in the world, they provide customers with access to products, insights, and experiences that enrich lives and build business success.

In this competition, you’ll apply your machine learning skills to predict credit default.
Specifically, you will leverage an industrial scale data set to build a machine learning model that challenges the current model in production.
Training, validation, and testing datasets include time-series behavioral data and anonymized customer profile information.
You're free to explore any technique to create the most powerful model, from creating features to using the data in a more organic way within a model.

If successful, you'll help create a better customer experience for cardholders by making it easier to be approved for a credit card.
Top solutions could challenge the credit default prediction model used by the world's largest payment card issuer—earning you cash prizes, the opportunity to interview with American Express, and potentially a rewarding new career.

# Evaluation

The evaluation metric, 𝑀, for this competition is the mean of two measures of rank ordering: Normalized Gini Coefficient, 𝐺, and default rate captured at 4%, 𝐷.

𝑀=0.5⋅(𝐺+𝐷)

The default rate captured at 4% is the percentage of the positive labels (defaults) captured within the highest-ranked 4% of the predictions, and represents a Sensitivity/Recall statistic.

For both of the sub-metrics 𝐺 and 𝐷, the negative labels are given a weight of 20 to adjust for downsampling.

This metric has a maximum value of 1.0.

Python code for calculating this metric can be found in this Notebook.

# Data

The objective of this competition is to predict the probability that a customer does not pay back their credit card balance amount in the future based on their monthly customer profile. The target binary variable is calculated by observing 18 months performance window after the latest credit card statement, and if the customer does not pay due amount in 120 days after their latest statement date it is considered a default event.

The dataset contains aggregated profile features for each customer at each statement date. Features are anonymized and normalized, and fall into the following general categories:

D_* = Delinquency variables
S_* = Spend variables
P_* = Payment variables
B_* = Balance variables
R_* = Risk variables
with the following features being categorical:

['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

Your task is to predict, for each customer_ID, the probability of a future payment default (target = 1).

Note that the negative class has been subsampled for this dataset at 5%, and thus receives a 20x weighting in the scoring metric.

## Files
train_data.csv - training data with multiple statement dates per customer_ID
train_labels.csv - target label for each customer_ID
test_data.csv - corresponding test data; your objective is to predict the target label for each customer_ID
sample_submission.csv - a sample submission file in the correct format
