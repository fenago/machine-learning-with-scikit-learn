
Chapter 8. Performance Evaluation Methods {.title}
-----------------------------------------

Your method of performance evaluation will vary by the type of machine
learning algorithm that you choose to implement. In general, there are
different metrics that can potentially determine how well your model is
performing at its given task for classification, regression, and
unsupervised machine learning algorithms. 

In this chapter, we will explore how the different performance
evaluation methods can help you to better understand your model. The
chapter will be split into three sections, as follows:

-   Performance evaluation for classification algorithms
-   Performance evaluation for regression algorithms 
-   Performance evaluation for unsupervised algorithms 


Technical requirements {.title style="clear: both"}
----------------------

* * * * *

You will be required to have Python 3.6 or greater, Pandas ≥
0.23.4, Scikit-learn ≥ 0.20.0, NumPy ≥ 1.15.1, Matplotlib ≥ 3.0.0,
and Scikit-plot ≥ 0.3.7installed on your system.

The code files of this chapter can be found on
GitHub:[https://github.com/PacktPublishing/Machine-Learning-with-scikit-learn-Quick-Start-Guide/blob/master/Chapter\_08.ipynb](https://github.com/PacktPublishing/Machine-Learning-with-scikit-learn-Quick-Start-Guide/blob/master/Chapter_08.ipynb)

Check out the following video to see the code in action:

[http://bit.ly/2EY4nJU](http://bit.ly/2EY4nJU)

 


Why is performance evaluation critical? {.title style="clear: both"}
---------------------------------------

* * * * *

It is key for you to understand why we need to evaluate the performance
of a model in the first place. Some of the potential reasons why
performance evaluation is critical are as follows:

-   **It prevents overfitting**:** **Overfitting occurs when your
    algorithm hugs the data too tightly and makes predictions that are
    specific to only one dataset. In other words, your model cannot
    generalize its predictions outside of the data that it was trained
    on.
-   **It prevents underfitting**:** **This is the exact opposite of
    overfitting. In this case, the model is very generic in nature.
-   **Understanding predictions**:** **Performance evaluation methods
    will help you to understand, in greater detail, how your model makes
    predictions, along with the nature of those predictions and other
    useful information, such as the accuracy of your model. 



Performance evaluation for classification algorithms {.title style="clear: both"}
----------------------------------------------------

* * * * *

In order to evaluate the performance of classification, let's consider
the two classification algorithms that we have built in this book:
k-nearest neighbors and logistic regression. 

The first step will be to implement both of these algorithms in the
fraud detection dataset. We can do this by using the following code:

Copy

``` {.programlisting .language-markup}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

#Reading in the fraud detection dataset 

df = pd.read_csv('fraud_prediction.csv')

#Creating the features 

features = df.drop('isFraud', axis = 1).values
target = df['isFraud'].values

#Splitting the data into training and test sets 

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42, stratify = target)

# Building the K-NN Classifier 

knn_classifier = KNeighborsClassifier(n_neighbors=3)

knn_classifier.fit(X_train, y_train)

#Initializing an logistic regression object

logistic_regression = linear_model.LogisticRegression()

#Fitting the model to the training and test sets

logistic_regression.fit(X_train, y_train)
```

In the preceding code, we read the fraud detection dataset into our
notebook and split the data into the features and target variables, as
usual. We then split the data into training and test sets, and build the
k-nearest neighbors and logistic regression models in the training data.

In this section, you will learn how to evaluate the performance of a
single model: k-nearest neighbors. You will also learn how to compare
and contrast multiple models. Therefore, you will learn about the
following things:

-   Confusion matrix
-   Normalized confusion matrix
-   Area under the curve (`auc`{.literal} score)
-   Cumulative gains curve 
-   Lift curve 
-   K-S statistic plot
-   Calibration plot
-   Learning curve
-   Cross-validated box plot

Some of the visualizations in this section will require a package titled
`scikit-plot`{.literal}. The `scikit-plot`{.literal} package is very
effective, and it is used to visualize the various performance measures
of machine learning models. It was specifically made for models that are
built using scikit-learn. 

 

In order to install `scikit-plot`{.literal} on your local machine, using
`pip`{.literal} in Terminal, we use the following code:

Copy

``` {.programlisting .language-markup}
pip3 install scikit-plot
```

If you are using the Anaconda distribution to manage your Python
packages, you can install `scikit-plot`{.literal} by using the following
code:

Copy

``` {.programlisting .language-markup}
conda install -c conda-forge scikit-plot
```

### The confusion matrix {.title}

Until now, we have used the accuracy as the sole measure of model
performance. That was fine, because we have a balanced dataset. A
balanced dataset is a dataset in which there are almost the same numbers
of labels for each category. In the dataset that we are working with,
8,000 labels belong to the fraudulent transactions, while 12,000 belong
to the non-fraudulent transactions. 

Imagine a situation in which 90% of our data had non-fraudulent
transactions, while only 10% of the transactions had fraudulent cases.
If the classifier reported an accuracy of 90%, it wouldn't make sense,
because most of the data that it has seen thus far were the
non-fraudulent cases and it has seen very little of the fraudulent
cases. So, even if it classified 90% of the cases accurately, it would
mean that most of the cases that it classified would belong to the
non-fraudulent cases. That would provide no value to us. 

A **confusion matrix** is a performance evaluation technique that can be
used in such cases, which do not involve a balanced dataset. The
confusion matrix for our dataset would look as follows:

![](./4_files/831ae8b3-6e3a-4222-8101-5c0a4ad5eec0.png)

Confusion matrix for fraudulent transactions 

The goal of the confusion matrix is to maximize the number of true
positives and true negatives, as this gives the correct predictions; it
also minimizes the number of false negatives and false positives, as
they give us the wrong predictions. 

Depending on your problem, the false positives might be more problematic
than the false negatives (and vice versa), and thus, the goal of
building the right classifier should be to solve your problem in the
best possible way. 

In order to implement the confusion matrix in scikit-learn, we use the
following code:

Copy

``` {.programlisting .language-markup}
from sklearn.metrics import confusion_matrix

#Creating predictions on the test set 

prediction = knn_classifier.predict(X_test)

#Creating the confusion matrix 

print(confusion_matrix(y_test, prediction))
```

This produces the following output:

![](./4_files/32ac34ff-9400-4574-a15b-7453c01828a5.png)

The confusion matrix output from our classifier for fraudulent
transactions 

In the preceding code, we create a set of predictions using
the `.predict()`{.literal}* *method on the test training data, and then
we use the `confusion_matrix()`{.literal}* *function on the test set of
the target variable and the predictions that were created earlier. 

The preceding confusion matrix looks almost perfect, as most cases are
classified into the true positive and true negative categories, along
the main diagonal. Only 46 cases are classified incorrectly, and this
number is almost equal. This means that the numbers of false positives
and false negatives are minimal and balanced, and one does not outweigh
the other. This is an example of the ideal classifier. 

Three other metrics that can be derived from the confusion matrix
are **precision**, **recall,** and **F1-score**. A high value of
precision indicates that not many non-fraudulent transactions are
classified as fraudulent, while a high value of recall indicates that
most of the fraudulent cases were predicted correctly. 

The F1-score is the weighted average of the precision and recall.

 

We can compute the precision and recall by using the following code:

Copy

``` {.programlisting .language-markup}
from sklearn.metrics import classification_report

#Creating the classification report 

print(classification_report(y_test, prediction))
```

This produces the following output:

![](./4_files/142c1079-567f-458b-bca9-082d28b651f7.png)

Classification report

In the preceding code, we use
the `classificiation_report()`{.literal} function with two arguments:
the test set of the target variable and the prediction variable that we
created for the confusion matrix earlier.

In the output, the precision, recall, and F1-score are all high, because
we have built the ideal machine learning model. These values range from
0 to 1, with 1 being the highest. 

### The normalized confusion matrix {.title}

A **normalized confusion matrix** makes it easier for the data scientist
to visually interpret how the labels are being predicted. In order to
construct a normalized confusion matrix, we use the following code:

Copy

``` {.programlisting .language-markup}
import matplotlib.pyplot as plt
import scikitplot as skplt

#Normalized confusion matrix for the K-NN model

prediction_labels = knn_classifier.predict(X_test)
skplt.metrics.plot_confusion_matrix(y_test, prediction_labels, normalize=True)
plt.show()
```

This results in the following normalized confusion matrix:

![](./4_files/2f33c0ab-feb8-4418-8e9c-56b2c6427bd6.png)

Normalized confusion matrix for the K-NN model

In the preceding plot, the predicted labels are along the *x *axis,
while the true (or actual) labels are along the *y *axis. We can see
that the model has 0.01, or 1%, of the predictions for the fraudulent
transactions incorrect, while 0.99, or 99%, of the fraudulent
transactions have been predicted correctly. We can also see that the
K-NN model predicted 100% of the non-fraudulent transactions correctly. 

Now, we can compare the performance of the logistic regression model by
using a normalized confusion matrix, as follows:

Copy

``` {.programlisting .language-markup}
#Normalized confusion matrix for the logistic regression model

prediction_labels = logistic_regression.predict(X_test)
skplt.metrics.plot_confusion_matrix(y_test, prediction_labels, normalize=True)
plt.show()
```

This results in the following normalized confusion matrix:

![](./4_files/44ca6956-98c8-4c05-b35e-5e0e38d68e04.png)

Normalized confusion matrix for the logistic regression model

In the preceding confusion matrix, it is clear that the logistic
regression model only predicted 42% of the non-fraudulent transactions
correctly. This indicates, almost instantly, that the k-nearest neighbor
model performed better.

### Area under the curve {.title}

The curve, in this case, is the **receiver operator characteristics**
(**ROC**) curve. This is a plot between the true positive rate and the
false positive rate. We can plot this curve as follows:

Copy

``` {.programlisting .language-markup}
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

#Probabilities for each prediction output 

target_prob = knn_classifier.predict_proba(X_test)[:,1]

#Plotting the ROC curve 

fpr, tpr, thresholds = roc_curve(y_test, target_prob)

plt.plot([0,1], [0,1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()
```

This produces the following curve:

![](./4_files/09d11506-51cd-4ef4-a95d-9604d77eae9c.png)

ROC curve 

In the preceding code, first, we create a set of probabilities for each
of the predicted labels. For instance, the predicted label of **`1`**
would have a certain set of probabilities associated with it, while the
label **`0`** would have a certain set of probabilities associated with
it. Using these probabilities, we use
the `roc_curve()`{.literal}* *function, along with the target test set,
to generate the ROC curve.

The preceding curve is an example of a perfect ROC curve. The preceding
curve has a true positive rate of 1.0, which indicates accurate
predictions, while it has a false positive rate of 0, which indicates a
lack of wrong predictions. 

Such a curve also has the most area under the curve, as compared to the
curves of models that have a lower accuracy. In order to compute the
area under the curve score, we use the following code:

Copy

``` {.programlisting .language-markup}
#Computing the auc score 

roc_auc_score(y_test, target_prob)
```

This produces a score of 0.99. A higher `auc`{.literal} score indicates
a better performing model. 

### Cumulative gains curve {.title}

When building multiple machine learning models, it is important to
understand which of the models in question produces the type of
predictions that you want it to generate. The **cumulative gains curve**
helps you with the process of model comparison, by telling you about the
percentage of a category/class that appears within a percentage of the
sample population for a particular model. 

In simple terms, in the fraud detection dataset, we might want to pick a
model that can predict a larger number of fraudulent transactions, as
opposed to a model that cannot. In order to construct the cumulative
gains plot for the k-nearest neighbors model, we use the following code:

Copy

``` {.programlisting .language-markup}
import scikitplot as skplt

target_prob = knn_classifier.predict_proba(X_test)
skplt.metrics.plot_cumulative_gain(y_test, target_prob)
plt.show()
```

This results in the following plot:

![](./4_files/74ec4a2a-240c-4676-94c5-f790dee06c03.png)

Cumulative gains plot for the k-nearest neighbors model

In the preceding code, the following applies:

-   First, we import the `scikit-plot`{.literal} package, which
    generates the preceding plot. We then compute the probabilities for
    the target variable, which, in this case, are the probabilities if a
    particular mobile transaction is fraudulent or not on the test data.
-   Finally, we use the `plot_cumulative_gain()`{.literal}* *function on
    these probabilities and the test data target labels, in order to
    generate the preceding plot.

How do we interpret the preceding plot? We simply look for the point at
which a certain percentage of the data contains 100% of the target
class. This is illustrated in the following diagram:

![](./4_files/b4caa494-7aa0-4e60-b9c2-baee1c30c51a.png)

Point at which 100% of the target class exists 

The point defined in the preceding diagram corresponds to a value of
0.3 on the *x *axis and 1.0 on the *y *axis. This means that 0.3 to 1.0
(or 30% to 100%) of the data will consist of the target class, 1, which
are the fraudulent transactions. 

This can also be interpreted as follows: 70% of the total data will
contain 100% of the fraudulent transaction predictions if you use the
k-nearest neighbors model.

Now, let's compute the cumulative gains curve for the logistic
regression model, and see if it is different. In order to do this, we
use the following code:

Copy

``` {.programlisting .language-markup}
#Cumulative gains plot for the logistic regression model

target_prob = logistic_regression.predict_proba(X_test)
skplt.metrics.plot_cumulative_gain(y_test, target_prob)
plt.show()
```

This results in the following plot:

![](./4_files/038ace2f-4368-4a0c-a024-cca439a7b2ac.png)

Cumulative gains plot for the logistic regression model

The preceding plot is similar to the cumulative gains plot that was
previously produced by the K-NN model, in that 70% of the data contains
100% of the target class. Therefore, using either the K-NN or the
logistic regression model will yield similar results. 

However, it is a good practice to compare how different models behave by
using the cumulative gains chart, in order to gain a fundamental
understanding of how your model makes predictions. 

### Lift curve {.title}

A **lift curve** gives you information about how well you can make
predictions by using a machine learning model, as opposed to when you
are not using one. In order to construct a lift curve for the k-nearest
neighbor model, we use the following code:

Copy

``` {.programlisting .language-markup}
# Lift curve for the K-NN model

target_prob = knn_classifier.predict_proba(X_test)
skplt.metrics.plot_lift_curve(y_test, target_prob)
plt.show()
```

This results in the following plot:

![](./4_files/f260676a-08a7-456c-94b3-78cf1a7b0615.png)

Lift curve for the K-NN model

How do we interpret the preceding lift curve? We have to look for the
point at which the curve dips. This is illustrated for you in the
following diagram:

![](./4_files/8ec7cbef-431c-4a48-b02f-116befb53685.png)

Point of interest in the lift curve

In the preceding plot, the point that is highlighted is the point that
we want to look for in any lift curve. The point tells us that 0.3, or
30%, of our total data will perform 3.5 times better when using the K-NN
predictive model, as opposed to when we do not use any model at all to
predict the fraudulent transactions. 

Now, we can construct the lift curve for the logistic regression model,
in order to compare and contrast the performance of the two models. We
can do this by using the following code:

Copy

``` {.programlisting .language-markup}
#Cumulative gains plot for the logistic regression model

target_prob = logistic_regression.predict_proba(X_test)
skplt.metrics.plot_lift_curve(y_test, target_prob)
plt.show()
```

This results in the following plot:

![](./4_files/e81ab108-790c-4bd2-a6e3-e2bd6eadda9c.png)

Lift curve for the logistic regression model

Although the plot tells us that 30% of the data will see an improved
performance (similar to that of the K-NN model that we built earlier in
order to predict the fraudulent transactions), there is a difference
when it comes to predicting the non-fraudulent transactions (the blue
line). 

For a small percentage of the data, the lift curve for the
non-fraudulent transactions is actually lower than the baseline (the
dotted line). This means that the logistic regression model does worse
than not using a predictive model for a small percentage of the data
when it comes to predicting the non-fraudulent transactions. 

### K-S statistic plot {.title}

The **K-S statistic plot**, or the **Kolmogorov Smirnov** statistic
plot, is a plot that tells you whether the model gets confused when it
comes to predicting the different labels in your dataset. In order to
illustrate what the term *confused* means in this case, we will
construct the K-S statistic plot for the K-NN model by using the
following code:

 

Copy

``` {.programlisting .language-markup}
#KS plot for the K-NN model

target_proba = knn_classifier.predict_proba(X_test)
skplt.metrics.plot_ks_statistic(y_test, target_proba)
plt.show()
```

This results in the following plot:

![](./4_files/02c063e0-0ff9-4f43-b27d-f52765fbb6e6.png)

K-S statistic plot for the K-NN model

In the preceding plot, the following applies:

-   The dotted line is the distance between the predictions for the
    fraudulent transactions (the yellow line at the bottom) and the
    non-fraudulent transactions (the blue line at the top). This
    distance is 0.985, as indicated by the plot.
-   A K-S statistic score that is close to 1 is usually a good
    indication that the model does not get confused between predicting
    the two different target labels, and can make a clear distinction
    when it comes to predicting the labels. 
-   In the preceding plot, the score of 0.985 can be observed as the
    difference between the two classes of predictions, for up to 70%
    (0.7) of the data. This can be observed along the *x *axis, as a
    threshold of 0.7 still has the maximum separation distance. 

We can now compute the K-S statistic plot for the logistic regression
model, in order to compare which of the two models provides a better
distinction in predictions between the two class labels. We can do this
by using the following code:

Copy

``` {.programlisting .language-markup}
#KS plot for the logistic regression model

target_proba = logistic_regression.predict_proba(X_test)
skplt.metrics.plot_ks_statistic(y_test, target_proba)
plt.show()
```

This results in the following plot:

![](./4_files/f08df259-4876-4b67-833d-1fbd5ac9cf30.png)

K-S statistic plot for the logistic regression model

Although the two models have the same separation score of 0.985, the
threshold at which the separation occurs is quite different. In the case
of logistic regression, this distance only occurs for the bottom 43% of
the data, since the maximum separation starts at a threshold of 0.57,
along the *x *axis. 

This means that the k-nearest neighbors model, which has a large
distance for about 70% of the total data, is much better at making
predictions about fraudulent transactions. 

### Calibration plot {.title}

A **calibration plot**, as the name suggests, tells you how well
calibrated your model is. A well-calibrated model will have a prediction
score equal to the fraction of the positive class (in this case, the
fraudulent transactions). In order to plot a calibration plot, we use
the following code:

Copy

``` {.programlisting .language-markup}
#Extracting the probabilites that the positive class will be predicted

knn_proba = knn_classifier.predict_proba(X_test)
log_proba = logistic_regression.predict_proba(X_test)

#Storing probabilities in a list

probas = [knn_proba, log_proba]

# Storing the model names in a list 

model_names = ["k_nn", "Logistic Regression"]

#Creating the calibration plot

skplt.metrics.plot_calibration_curve(y_test, probas, model_names)

plt.show()
```

 

 

 

 

This results in the following calibration plot:

![](./4_files/998a0554-ca8a-4aa2-8298-5069e84fbfad.png)

Calibration plot for the two models 

In the preceding code, the following applies:

1.  First, we compute the probability that the positive class
    (fraudulent transactions) will be predicted for each model. 
2.  Then, we store these probabilities and the model names in a list. 
3.  Finally, we use the `plot_calibration_curve()`{.literal}* *function
    from the `scikit-plot`{.literal} package with these probabilities,
    the test labels, and the model names, in order to create the
    calibration plot. 

This results in the preceding calibration plot, which can be explained
as follows:

-   The dotted line represents the perfect calibration plot. This is
    because the mean prediction value has the exact value of the
    fraction of the positive class at each and every point. 
-   From the plot, it is clear that the k-nearest neighbors model is
    much better calibrated than the calibration plot of the logistic
    regression model.
-   This is because the calibration plot of the k-nearest neighbors
    model follows that of the ideal calibration plot much more closely
    than the calibration plot of the logistic regression model. 

 

### Learning curve {.title}

A **learning curve** is a plot that compares how the training accuracy
scores and the test accuracy scores vary as the number of samples/rows
added to the data increases. In order to construct the learning curve
for the k-nearest neighbors model, we use the following code:

Copy

``` {.programlisting .language-markup}
skplt.estimators.plot_learning_curve(knn_classifier, features, target)

plt.show()
```

This results in the following plot:

![](./4_files/2b6b49fe-589a-4855-81c6-46a6e34bd2ad.png)

Learning curve for the K-NN model

In the preceding curve, the following applies:

1.  The training score and the test score are only the highest when the
    number of samples is 15,000. This suggests that even if we had only
    15,000 samples (instead of the 17,500), we would still get the best
    possible results. 
2.  Anything under the 15,000 samples will mean that the test
    cross-validated scores will be much lower than the training scores,
    suggesting that the model is overfit.

 

### Cross-validated box plot {.title}

In this plot, we compare the cross-validated accuracy scores of multiple
models by making use of box plots. In order to do so, we use the
following code:

Copy

``` {.programlisting .language-markup}
from sklearn import model_selection

#List of models

models = [('k-NN', knn_classifier), ('LR', logistic_regression)]

#Initializing empty lists in order to store the results
cv_scores = []
model_name_list = []

for name, model in models:

    #5-fold cross validation
    cv_5 = model_selection.KFold(n_splits= 5, random_state= 50)
    # Evaluating the accuracy scores
    cv_score = model_selection.cross_val_score(model, X_test, y_test, cv = cv_5, scoring= 'accuracy')
    cv_scores.append(cv_score)
    model_name_list.append(name)

# Plotting the cross-validated box plot 

fig = plt.figure()
fig.suptitle('Boxplot of 5-fold cross validated scores for all the models')
ax = fig.add_subplot(111)
plt.boxplot(cv_scores)
ax.set_xticklabels(model_name_list)
plt.show()
```

 

 

 

 

This results in the following plot:

![](./4_files/55df8d01-1a83-4dce-8eab-a75702696d44.png)

Cross-validated box plot

In the preceding code, the following applies:

1.  First, we store the models that we want to compare in a list. 
2.  Then, we initialize two empty lists, in order to store the results
    of the cross-validated accuracy scores and the names of the models,
    so that we can use them later, in order to create the box plots. 
3.  We then iterate over each model in the list of models, and use
    the `model_selection.KFold()`{.literal}* *function in order to split
    the data into a five-fold cross-validated set.
4.  Next, we extract the five-fold cross-validated scores by using
    the `model_selection.cross_val_scores()`{.literal}* *function and
    append the scores, along with the model names, into the lists that
    we initialized at the beginning of the code. 
5.  Finally, a box plot is created, displaying the cross-validated
    scores in a box plot. 

The list that we created consists of the five cross-validated scores,
along with the model names. A box plot takes these five scores for each
model and computes the min, max, median, first, and third quartiles, in
the form of a box plot. 

 

In the preceding plot, the following applies:

1.  It is clear that the K-NN model has the highest value of accuracy,
    with the lowest difference between the minimum and maximum values.
2.  The logistic regression model, on the other hand, has the greatest
    difference between the minimum and maximum values, and has an
    outlier in its accuracy score, as well.



Performance evaluation for regression algorithms {.title style="clear: both"}
------------------------------------------------

* * * * *

There are three main metrics that you can use to evaluate the
performance of the regression algorithm that you built, as follows:

-   **Mean absolute error** (**MAE**)
-   **Mean squared error** (**MSE**)
-   **Root mean squared error** (**RMSE**)

In this section, you will learn what the three metrics are, how they
work, and how you can implement them using scikit-learn. The first step
is to build the linear regression algorithm. We can do this by using the
following code:

Copy

``` {.programlisting .language-markup}
## Building a simple linear regression model

#Reading in the dataset

df = pd.read_csv('fraud_prediction.csv')

#Define the feature and target arrays

feature = df['oldbalanceOrg'].values
target = df['amount'].values

#Initializing a linear regression model 

linear_reg = linear_model.LinearRegression()

#Reshaping the array since we only have a single feature

feature = feature.reshape(-1, 1)
target = target.reshape(-1, 1)

#Fitting the model on the data

linear_reg.fit(feature, target)

predictions = linear_reg.predict(feature)
```

### Mean absolute error {.title}

The mean absolute error is given by the following formula:

![](./5_files/33c56bea-c4e1-4da9-b913-a418654e84bd.png)

MAE formula

In the preceding formula, 

![](./5_files/3242a2a3-8697-42d5-a15e-1b54e6be57c3.png)

 represents the true (or actual) value of the output, while the 

![](./5_files/78a8729c-1f1a-43cd-beb5-79bb25c6bdac.png)

 hat represents the predicted output values. Therefore, by computing the
summation of the difference between the true value and the predicted
value of the output for each row in your data, and then dividing it by
the total number of observations, you get the mean value of the absolute
error. 

In order to implement the MAE in scikit-learn, we use the following
code:

Copy

``` {.programlisting .language-markup}
from sklearn import metrics

metrics.mean_absolute_error(target, predictions)
```

In the preceding code, the `mean_absolute_error()`{.literal}* *function
from the `metrics`{.literal} module in scikit-learn is used to compute
the MAE. It takes in two arguments: the real/true output, which is the
target, and the predictions, which are the predicted outputs. 

### Mean squared error {.title}

The mean squared error is given by the following formula:

![](./5_files/b42c83e6-1b47-43e6-9fcc-98d05c81cf12.png)

MSE formula

The preceding formula is similar to the formula that we saw for the mean
absolute error, except that instead of computing the absolute difference
between the true and predicted output values, we compute the square of
the difference. 

In order to implement the MSE in scikit-learn, we use the following
code:

Copy

``` {.programlisting .language-markup}
metrics.mean_squared_error(target, predictions)
```

We use the `mean_squared_error()`{.literal}* *function from the
`metrics`{.literal} module, with the real/true output values and the
predictions as arguments. The mean squared error is better at detecting
larger errors, because we square the errors, instead of depending on
only the difference. 

### Root mean squared error {.title}

The root mean squared error is given by the following formula:

![](./5_files/30f64614-99d6-41f1-a2b0-e8eea811c10d.png)

The preceding formula is very similar to that of the mean squared error,
except for the fact that we take the square root of the MSE formula. 

In order to compute the RMSE in scikit-learn, we use the following code:

Copy

``` {.programlisting .language-markup}
import numpy as np

np.sqrt(metrics.mean_squared_error(target, predictions))
```

In the preceding code, we use
the `mean_squared_error()`{.literal}* *function with the true/real
output and the predictions, and then we take the square root of this
answer by using the `np.sqrt()`{.literal}* *function from the
`numpy`{.literal} package. 

Compared to the MAE and the MSE, the RMSE is the best possible metric
that you can use in order to evaluate the linear regression model, since
this detects large errors and gives you the value in terms of the output
units. The key takeaway from using any one of the three metrics is that
the value that these `metrics`{.literal} gives you should be as low as
possible, indicating that the model has relatively low error values. 



Performance evaluation for unsupervised algorithms {.title style="clear: both"}
--------------------------------------------------

* * * * *

In this section, you will learn how to evaluate the performance of an
unsupervised machine learning algorithm, such as the k-means algorithm.
The first step is to build a simple k-means model. We can do so by using
the following code:

Copy

``` {.programlisting .language-markup}
#Reading in the dataset

df = pd.read_csv('fraud_prediction.csv')

#Dropping the target feature & the index

df = df.drop(['Unnamed: 0', 'isFraud'], axis = 1)

#Initializing K-means with 2 clusters

k_means = KMeans(n_clusters = 2)
```

Now that we have a simple k-means model with two clusters, we can
proceed to evaluate the model's performance. The different visual
performance charts that can be deployed are as follows:

-   Elbow plot
-   Silhouette analysis plot

In this section, you will learn how to create and interpret each of the
preceding plots. 

### Elbow plot {.title}

In order to construct an elbow plot, we use the following code:

Copy

``` {.programlisting .language-markup}
skplt.cluster.plot_elbow_curve(k_means, df, cluster_ranges=range(1, 20))
plt.show()
```

This results in the following plot:

![](./6_files/4e8c2c99-8bd9-4fd2-aef4-a2122367a7d8.png)

Elbow plot

The elbow plot is a plot between the number of clusters that the model
takes into consideration along the *x *axis and the sum of the squared
errors along the *y *axis.

In the preceding code, the following applies:

-   We use the `plot_elbow_curve()`{.literal}* *function with the
    k-means model, the data, and the number of clusters that we want to
    evaluate
-   In this case, we define a range of 1 to 19 clusters

In the preceding plot, the following applies:

-   It is clear that the elbow point, or the point at which the sum of
    the squared errors (*y *axis) starts decreasing very slowly, is
    where the number of clusters is 4.
-   The plot also gives you another interesting metric on the *y *axis
    (right-hand side), which is the clustering duration (in seconds).
    This indicates the amount of time it took for the algorithm to
    create the clusters, in seconds.

 

Summary {.title style="clear: both"}
-------

* * * * *

In this chapter, you learned how to evaluate the performances of the
three different types of machine learning algorithms: classification,
regression, and unsupervised. 

For the classification algorithms, you learned how to evaluate the
performance of a model by using a series of visual techniques, such as
the confusion matrix, normalized confusion matrix, area under the curve,
K-S statistic plot, cumulative gains plot, lift curve, calibration plot,
learning curve, and cross-validated box plot. 

For the regression algorithms, you learned how to evaluate the
performance of a model by using three metrics: the mean squared error,
mean absolute error, and root mean squared error.

Finally, for the unsupervised machine learning algorithms, you learned
how to evaluate the performance of a model by using the elbow plot. 

Congratulations! You have now made it to the end of your machine
learning journey with scikit-learn. You've made your way through eight
chapters, which gave you the quickest entry point into the wonderful
world of machine learning with one of the world's most popular machine
learning frameworks: scikit-learn. 

In this book, you learned about the following topics:

-   What machine learning is (in a nutshell) and the different types and
    applications of machine learning
-   Supervised machine learning algorithms, such as K-NN, logistic
    regression, Naive Bayes, support vector machines, and linear
    regression
-   Unsupervised machine learning algorithms, such as the k-means
    algorithm
-   Algorithms that can perform both classification and regression, such
    as decision trees, random forests, and gradient-boosted trees

I hope that you can make the best possible use of the application based
on the knowledge that this book has given you, allowing you to solve
many real-world problems by using machine learning as your tool!

