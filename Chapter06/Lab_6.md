
Chapter 6. Classification and Regression with Trees
---------------------------------------------------

Tree based algorithms are very popular for two reasons: they are
interpretable, and they make sound predictions that have won many
machine learning competitions on online platforms, such as Kaggle.
Furthermore, they have many use cases outside of machine learning for
solving problems, both simple and complex.

Building a tree is an approach to decision-making used in almost all
industries. Trees can be used to solve both classification- and
regression-based problems, and have several use cases that make them the
go-to solution!

This chapter is broadly divided into the following two sections:

-   Classification trees
-   Regression trees

Each section will cover the fundamental theory of different types of
tree based algorithms, along with their implementation in scikit-learn.
By the end of this chapter, you will have learned how to aggregate
several algorithms into an **ensemble** and have them vote on what the
best prediction is.


Technical requirements
----------------------

* * * * *

You will be required to have Python 3.6 or greater, Pandas ≥
0.23.4, Scikit-learn ≥ 0.20.0, and Matplotlib ≥ 3.0.0 installed on your
system.

The code files of this chapter can be found on
GitHub:[https://github.com/PacktPublishing/Machine-Learning-with-scikit-learn-Quick-Start-Guide/blob/master/Chapter\_06.ipynb](https://github.com/PacktPublishing/Machine-Learning-with-scikit-learn-Quick-Start-Guide/blob/master/Chapter_06.ipynb)[.](https://github.com/PacktPublishing/Machine-Learning-with-scikit-learn-Quick-Start-Guide/blob/master/Chapter_06.ipynb)

 

Check out the following video to see the code in action:

[http://bit.ly/2SrPP7R](http://bit.ly/2SrPP7R)



Classification trees
--------------------

* * * * *

Classification trees are used to predict a category or class. This is
similar to the classification algorithms that you have learned about
previouslyin this book, such as the k-nearest neighbors algorithm or
logistic regression.

Broadly speaking, there are three tree based algorithms that are used to
solve classification problems:

-   The decision tree classifier
-   The random forest classifier
-   The AdaBoost classifier

In this section, you will learn how each of these tree based algorithms
works, in order to classify a row of data as a particular class or
category.

### The decision tree classifier

The decision tree is the simplest tree based algorithm, and serves as
the foundation for the other two algorithms. Let's consider the
following simple decision tree:

![](./3_files/1f99278c-4182-4d1a-b104-e042c6f93ce9.png)

A simple decision tree

A decision tree, in simple terms, is a set of rules that help us
classify observations into distinct groups. In the previous diagram, the
rule could be written as the following:

Copy

``` {.programlisting .language-markup}
If (value of feature is less than 50); then (put the triangles in the left-hand box and put the circles in the right-hand box).
```

The preceding decision tree perfectly divides the observations into two
distinct groups. This is a characteristic of the ideal decision tree.
The first box on the top is called the **root** of the tree, and is the
most important feature of the tree when it comes to deciding how to
group the observations.

The boxes under the root node are known as the **children**. In the
preceding tree, the **children** are also the **leaf** nodes. The
**leaf** is the last set of boxes, usually in the bottommost part of the
tree. As you might have guessed, the decision tree represents a regular
tree, but inverted, or upside down.

#### Picking the best feature

How does the decision tree decide which feature is the best? The best
feature is one that offers the best possible split, and divides the tree
into two or more distinct groups, depending on the number of classes or
categories that we have in the data. Let's have a look at the following
diagram:

![](./3_files/ce6d6aaa-33fc-4e14-b705-2f86fd2db0f0.png)

A decision tree showing a good split

In the preceding diagram, the following happens:

1.  The tree splits the data from the root node into two distinct
    groups.
2.  In the left-hand group, we see that there are two triangles and one
    circle.
3.  In the right-hand group, we see that there are two circles and one
    triangle.
4.  Since the tree got the majority of each class into one group, we can
    say that the tree has done a good job when it comes to splitting the
    data into distinct groups.

Let's take a look at another example—this time, one in which the split
is bad. Consider the following diagram:

![](./3_files/1bf114e1-67ff-4123-9b93-e914271b7f86.png)

A decision tree with a bad split

In the preceding diagram, the following happens:

1.  The tree splits the data in the root node into four distinct groups.
    This is bad in itself, as it is clear that there are only two
    categories (circle and triangle).
2.  Furthermore, each group has one triangle and one circle.
3.  There is no majority class or category in any one of the four
    groups. Each group has 50% of one category; therefore, the tree
    cannot come to a conclusive decision, unless it relies on more
    features, which then increases the complexity of the tree.

#### The Gini coefficient

The metric that the decision tree uses to decide if the root node is
called the *Gini coefficient*. The higher the value of this coefficient,
the better the job that this particular feature does at splitting the
data into distinct groups. In order to learn how to compute the Gini
coefficient for a feature, let's consider the following diagram:

![](./3_files/ce6d6aaa-33fc-4e14-b705-2f86fd2db0f0.png)

Computing the Gini coefficient

In the preceding diagram, the following happens:

1.  The feature splits the data into two groups.
2.  In the left-hand group, we have two triangles and one circle.
3.  Therefore, the Gini for the left-hand group is (2 triangles/3 total
    data points)\^2+ (1 circle/3 total data points)\^2.
4.  To calculate this, do the following:
    ![](./3_files/e71b0785-d3f0-4207-ba17-9fc99937bac5.png)
    0.55.
5.  A value of 0.55 for the Gini coefficient indicates that the root of
    this tree splits the data in such a way that each group has a
    majority category.
6.  A perfect root feature would have a Gini coefficient of 1. This
    means that each group has only one class/category.
7.  A bad root feature would have a Gini coefficient of 0.5, which
    indicates that there is no distinct class/category in a group.

In reality, the decision tree is built in a recursive manner, with the
tree picking a random attribute for the root and then computing the Gini
coefficient for that attribute. It does this until it finds the
attribute that best splits the data in a node into groups that have
distinct classes and categories.

#### Implementing the decision tree classifier in scikit-learn

In this section, you will learn how to implement the decision tree
classifier in scikit-learn. We will work with the same fraud detection
dataset. The first step is to load the dataset into the Jupyter
Notebook. We can do this by using the following code:

Copy

``` {.programlisting .language-markup}
import pandas as pd

df = pd.read_csv('fraud_prediction.csv')
```

The next step is to split the data into training and test sets. We can
do this using the following code:

Copy

``` {.programlisting .language-markup}
#Creating the features 

features = df.drop('isFraud', axis = 1).values
target = df['isFraud'].values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42, stratify = target)
```

We can now build the initial decision tree classifier on the training
data, and test its accuracy on the test data, by using the following
code:

Copy

``` {.programlisting .language-markup}
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = 'gini', random_state = 50)

#Fitting on the training data

dt.fit(X_train, y_train)

#Testing accuracy on the test data

dt.score(X_test, y_test)
```

 

 

 

 

 

 

In the preceding code, we do the following:

1.  First, we import `DecisionTreeClassifier`{.literal} from
    scikit-learn.
2.  We then initialize a `DecisionTreeClassifier`{.literal} object with
    two arguments. The first, `criterion`{.literal}, is the metric with
    which the tree picks the most important features in a recursive
    manner, which, in this case, is the Gini coefficient. The second is
    `random_state`{.literal}, which is set to 50 so that the model
    produces the same result every time we run it.
3.  Finally, we fit the model on the training data and evaluate its
    accuracy on the test data.

#### Hyperparameter tuning for the decision tree

The decision tree has a plethora of hyperparameters that require
fine-tuning in order to derive the best possible model that reduces the
generalization error as much as possible. In this section, we will focus
on two specific hyperparameters:

-   **Max depth**: This is the maximum number of children nodes that can
    grow out from the decision tree until the tree is cut off. For
    example, if this is set to 3, then the tree will use three children
    nodes and cut the tree off before it can grow any more.
-   **Min samples leaf:** This is the minimum number of samples, or data
    points, that are required to be present in the leaf node. The leaf
    node is the last node of the tree. If this parameter is, for
    example, set to a value of 0.04, it tells the tree that it must grow
    until the last node contains 4% of the total samples in the data.

In order to optimize the ideal hyperparameter and to extract the best
possible decision tree, we use the `GridSearchCV`{.literal} module from
scikit-learn. We can set this up using the following code:

Copy

``` {.programlisting .language-markup}
from sklearn.model_selection import GridSearchCV

#Creating a grid of different hyperparameters

grid_params = {
    'max_depth': [1,2,3,4,5,6],
    'min_samples_leaf': [0.02,0.04, 0.06, 0.08]
}

#Building a 10 fold Cross Validated GridSearchCV object

grid_object = GridSearchCV(estimator = dt, param_grid = grid_params, scoring = 'accuracy', cv = 10, n_jobs = -1)
```

 

In the preceding code, we do the following:

1.  We first import the `GridSearchCV`{.literal} module from
    scikit-learn.
2.  Next, we create a dictionary of possible values for the
    hyperparameters and store it as `grid_params`{.literal}.
3.  Finally, we create a `GridSearchCV`{.literal} object with the
    decision tree classifier as the estimator; that is, the dictionary
    of hyperparameter values.
4.  We set the `scoring`{.literal} argument as `accuracy`{.literal},
    since we want to extract the accuracy of the best model found by
    `GridSearchCV`{.literal}.

We then fit this grid object to the training data using the following
code:

Copy

``` {.programlisting .language-markup}
#Fitting the grid to the training data

grid_object.fit(X_train, y_train)
```

We can then extract the best set of parameters using the following code:

Copy

``` {.programlisting .language-markup}
#Extracting the best parameters

grid_object.best_params_
```

The output of the preceding code indicates that a maximum depth of 1 and
a minimum number of samples at the leaf node of 0.02 are the best
parameters for this data. We can use these optimal parameters and
construct a new decision tree using thefollowingcode:

Copy

``` {.programlisting .language-markup}
#Extracting the best parameters

grid_object.best_params_
```

#### Visualizing the decision tree

One of the best aspects of building and implementing a decision tree in
order to solve problems is that it can be interpreted quite easily,
using a decision tree diagram that explains how the algorithm that you
built works. In order to visualize a simple decision tree for the fraud
detection dataset, we use thefollowingcode:

Copy

``` {.programlisting .language-markup}
#Package requirements 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import tree
```

We start by importing the required packages. The new packages here are
the following:

-   `StringIO`{.literal}
-   `Image`{.literal}
-   `export_graphviz`{.literal}
-   `pydotplus`{.literal}
-   `tree`{.literal}

The installations of the packages were covered in [Chapter
1](https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789343700/1),
*Introducing Machine Learning with scikit-learn*.

Then, we read in the dataset and initialize a decision tree classifier,
as shown in thefollowingcode:

Copy

``` {.programlisting .language-markup}
#Reading in the data

df = pd.read_csv('fraud_prediction.csv')
df = df.drop(['Unnamed: 0'], axis = 1)

#Creating the features 

features = df.drop('isFraud', axis = 1).values
target = df['isFraud'].values

#Initializing the DT classifier

dt = DecisionTreeClassifier(criterion = 'gini', random_state = 50, max_depth= 5)
```

Next, we fit the tree on the features and target, and then extract the
feature names separately:

Copy

``` {.programlisting .language-markup}
#Fitting the classifier on the data

dt.fit(features, target)

#Extracting the feature names

feature_names = df.drop('isFraud', axis = 1)
```

We can then visualize the decision tree using thefollowingcode:

Copy

``` {.programlisting .language-markup}
#Creating the tree visualization

data = tree.export_graphviz(dt, out_file=None, feature_names= feature_names.columns.values, proportion= True)

graph = pydotplus.graph_from_dot_data(data) 

# Show graph
Image(graph.create_png())
```

In the preceding code, we do the following:

1.  We use the `tree.export_graphviz()`{.literal} function in order to
    construct the decision tree object, and store it in a variable
    called `data`{.literal}.
2.  This function uses a couple of arguments: `dt`{.literal} is the
    decision tree that you built;`out_file`{.literal} is set to
    `None`{.literal}, as we do not want to send the tree visualization
    to any file outside our Jupyter Notebook; the
    `feature_names`{.literal} are those we defined earlier; and
    `proportion`{.literal} is, set to `True`{.literal} (this will be
    explained in more detail later).
3.  We then construct a graph of the data contained within the tree so
    that we can visualize this decision tree graph by using the
    `pydotplus. graph_from_dot_data()`{.literal} function on the
    `data`{.literal} variable, which contains data about the decision
    tree.
4.  Finally, we visualize the decision tree using the
    `Image()`{.literal} function, by passing the graph of the decision
    tree to it.

This results in a decision tree like that illustrated in the following
diagram:

![](./3_files/b6203ea3-7a16-4cc7-85fc-d89933eacc54.png)

The resultant decision tree

The tree might seem pretty complex to interpret at first, but it's not!
In order to interpret this tree, let's consider the root node and the
first two children only. This is illustrated in the following diagram:

![](./3_files/58622285-a445-442c-944a-8a4a43efb88c.png)

A snippet of the decision tree

In the preceding diagram, note the following:

-   In the root node, the tree has identified the 'step' feature as the
    feature with the highest Gini value.
-   The root node makes the split in such a way that 0.71, or 71%, of
    the data falls into the non-fraudulent transactions, while 0.29, or
    29%, of the transactions fall into the fraudulent transactions
    category.
-   If the step is greater than or equal to 7.5 (the right-hand side),
    then all of the transactions are classified as fraudulent.
-   If the step is less than or equal to 7.5 (the left-hand side), then
    0.996, or 99.6%, of the transactions are classified as
    non-fraudulent, while 0.004, or 0.4%, of the transactions are
    classified as fraudulent.
-   If the amount is greater than or equal to 4,618,196.0, then all of
    the transactions are classified as fraudulent.
-   If the amount is less than or equal to 4,618,196.0, then 0.996, or
    99.6%, of the transactions are classified as non-fraudulent, while
    0.004, or 0.4%, of the transactions are classified as fraudulent.

Note how the decision tree is simply a set of If-then rules, constructed
in a nested manner.

### The random forests classifier

Now that you understand the core principles of the decision tree at a
very foundational level, we will next explore what random forests are.
Random forests are a form of *ensemble* learning. An ensemble learning
method is one that makes use of multiple machine learning models to make
a decision.

Let's consider the following diagram:

![](./3_files/fc7c9ae3-9124-40e5-a150-b15f450ed0a6.png)

The concept of ensemble learning

 

The random forest algorithm operates as follows:

1.  Assume that you initially have a dataset with 100 features.
2.  From this, we will build a decision tree with 10 features initially.
    The features are selected randomly.
3.  Now, using a random selection of the remaining 90 features, we
    construct the next decision tree, again with 10 features.
4.  This process continues until there are no more features left to
    build a decision tree with.
5.  At this point in time, we have 10 decision trees, each with 10
    features.
6.  Each decision tree is known as the **base estimator** of the random
    forest.
7.  Thus, we have a forest of trees, each built using a random set of 10
    features.

The next step for the algorithm is to make the prediction. In order to
better understand how the random forest algorithm makes predictions,
consider the following diagram:

![](./3_files/8f837ee0-5b91-404b-a2b1-6d9c23c70aab.png)

The process of making predictions in random forests

In the preceding diagram, the following occurs:

1.  Let's assume that there are 10 decision trees in the random forest.
2.  Each decision tree makes a single prediction for the data that comes
    in.
3.  If six trees predict class A, and four trees predict class B, then
    the final prediction of the random forest algorithm is class A, as
    it had the majority vote.
4.  This process of voting on a prediction, based on the outputs of
    multiple models, is known as ensemble learning.

Now that you have learned how the algorithm works internally, we can
implement it using scikit-learn!

#### Implementing the random forest classifier in scikit-learn

In this section, we will implement the random forest classifier in
scikit-learn. The first step is to read in the data, and split it into
training and test sets. This can be done by using the following code:

Copy

``` {.programlisting .language-markup}
import pandas as pd

#Reading in the dataset

df = pd.read_csv('fraud_prediction.csv')

#Dropping the index

df = df.drop(['Unnamed: 0'], axis = 1)

#Creating the features 

features = df.drop('isFraud', axis = 1).values
target = df['isFraud'].values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42, stratify = target)
```

The next step is to build the random forest classifier. We can do that
using the following code:

Copy

``` {.programlisting .language-markup}
from sklearn.ensemble import RandomForestClassifier

#Initiliazing an Random Forest Classifier with default parameters

rf_classifier = RandomForestClassifier(random_state = 50)

#Fitting the classifier on the training data

rf_classifier.fit(X_train, y_train)

#Extracting the scores

rf_classifier.score(X_test, y_test)
```

In the preceding code block, we do the following:

1.  We first import `RandomForestClassifier`{.literal} from
    scikit-learn.
2.  Next, we initialize a random forest classifier model.
3.  We then fit this model to our training data, and evaluate its
    accuracy on the test data.

#### Hyperparameter tuning for random forest algorithms

In this section, we will learn how to optimize the hyperparameters of
the random forest algorithm. Since random forests are fundamentally
based on multiple decision trees, the hyperparameters are very similar.
In order to optimize the hyperparameters, we use the following code:

Copy

``` {.programlisting .language-markup}
from sklearn.model_selection import GridSearchCV

#Creating a grid of different hyperparameters

grid_params = {
 'n_estimators': [100,200, 300,400,5000],
 'max_depth': [1,2,4,6,8],
 'min_samples_leaf': [0.05, 0.1, 0.2]
}

#Building a 3 fold Cross-Validated GridSearchCV object

grid_object = GridSearchCV(estimator = rf_classifier, param_grid = grid_params, scoring = 'accuracy', cv = 3, n_jobs = -1)

#Fitting the grid to the training data

grid_object.fit(X_train, y_train)

#Extracting the best parameters

grid_object.bestparams

#Extracting the best model

rf_best = grid_object.bestestimator_
```

In the preceding code block, we do the following:

1.  We first import the `GridSearchCV`{.literal} package.
2.  We initialize a dictionary of hyperparameter values. The
    `max_depth`{.literal} and `min_samples_leaf`{.literal} values are
    similar to those of the decision tree.
3.  However, `n_estimators`{.literal} is a new parameter, covering the
    total number of trees that you want your random forest algorithm to
    consider while making the final prediction.
4.  We then build and fit the `gridsearch`{.literal} object to the
    training data and extract the optimal parameters.
5.  The best model is then extracted using these optimal
    hyperparameters.

### The AdaBoost classifier

In the section, you will learn how the AdaBoost classifier works
internally, and how the concept of boosting might be used to give you
better results. Boosting is a form of ensemble machine learning, in
which a machine learning model learns from the mistakes of the models
that were previously built, thereby increasing its final prediction
accuracy.

AdaBoost stands for Adaptive Boosting, and is a boosting algorithm in
which a lot of importance is given to the rows of data that the initial
predictive model got wrong. This ensures that the next predictive model
will not make the same mistakes.

The process by which the AdaBoost algorithm works is illustrated in the
following diagram:

![](./3_files/e3126163-d7d6-4584-99d6-798a12662965.png)

An outline of the AdaBoost algorithm

In the preceding diagram of the AdaBoost algorithm, the following
happens:

1.  The first decision tree is built and outputs a set of predictions.
2.  The predictions that the first decision tree got wrong are given a
    weight of `w`{.literal}. This means that, if the weight is set to 2,
    then two instances of that particular sample are introduced into the
    dataset.
3.  This enables decision tree 2 to learn at a faster rate, since we
    have more samples of the data in which an error was made beforehand.
4.  This process is repeated until all the trees are built.
5.  Finally, the predictions of all the trees are gathered, and a
    weighted vote is initiated in order to determine the final
    prediction.

#### Implementing the AdaBoost classifier in scikit-learn

In this section, we will learn how we can implement the AdaBoost
classifier in scikit-learn in order to predict if a transaction is
fraudulent or not. As usual, the first step is to import the data and
split it into training and testing sets.

This can be done with the following code:

Copy

``` {.programlisting .language-markup}
#Reading in the dataset

df = pd.read_csv('fraud_prediction.csv')

#Dropping the index

df = df.drop(['Unnamed: 0'], axis = 1)

#Creating the features 

features = df.drop('isFraud', axis = 1).values
target = df['isFraud'].values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42, stratify = target)
```

The next step is to build the AdaBoost classifier. We can do this using
the following code:

Copy

``` {.programlisting .language-markup}
from sklearn.ensemble import AdaBoostClassifier

#Initialize a tree (Decision Tree with max depth = 1)

tree = DecisionTreeClassifier(max_depth=1, random_state = 42)

#Initialize an AdaBoost classifier with the tree as the base estimator

ada_boost = AdaBoostClassifier(base_estimator = tree, n_estimators=100)

#Fitting the AdaBoost classifier to the training set

ada_boost.fit(X_train, y_train)

#Extracting the accuracy scores from the classifier

ada_boost.score(X_test, y_test)
```

In the preceding code block, we do the following:

1.  We first import the `AdaBoostClassifier`{.literal} package from
    scikit-learn.
2.  Next, we initialize a decision tree that forms the base of our
    AdaBoost classifier.
3.  We then build the AdaBoost classifier, with the base estimator as
    the decision tree, and we specify that we want 100 decision trees in
    total.
4.  Finally, we fit the classifier to the training data, and extract the
    accuracy scores from the test data.

#### Hyperparameter tuning for the AdaBoost classifier

In this section, we will learn how to tune the hyperparameters of the
AdaBoost classifier. The AdaBoost classifier has only one parameter of
interest—the number of base estimators, or decision trees.

We can optimize the hyperparameters of the AdaBoost classifier using the
following code:

Copy

``` {.programlisting .language-markup}
from sklearn.model_selection import GridSearchCV

#Creating a grid of hyperparameters

grid_params = {
    'n_estimators': [100,200,300]
}

#Building a 3 fold CV GridSearchCV object

grid_object = GridSearchCV(estimator = ada_boost, param_grid = grid_params, scoring = 'accuracy', cv = 3, n_jobs = -1)

#Fitting the grid to the training data

grid_object.fit(X_train, y_train)

#Extracting the best parameters

grid_object.bestparams

#Extracting the best model

ada_best = grid_object.best_estimator_
```

In the preceding code, we do the following:

1.  We first import the `GridSearchCV`{.literal} package.
2.  We initialize a dictionary of hyperparameter values. In this case,
    `n_estimators`{.literal} is the number of decision trees.
3.  We then build and fit the `gridsearch`{.literal} object to the
    training data and extract the best parameters.
4.  The best model is then extracted using these optimal
    hyperparameters.



Regression trees
----------------

* * * * *

You have learned how trees are used in order to classify a prediction as
belonging to a particular class or category. However, trees can also be
used to solve problems related to predicting numeric outcomes. In this
section, you will learn about the three types of tree based algorithms
that you can implement in scikit-learn in order to predict numeric
outcomes, instead of classes:

-   The decision tree regressor
-   The random forest regressor
-   The gradient boosted tree

### The decision tree regressor

When we have data that is non-linear in nature, a linear regression
model might not be the best model to choose. In such situations, it
makes sense to choose a model that can fully capture the non-linearity
of such data. A decision tree regressor can be used to predict numeric
outcomes, just like that of the linear regression model.

In the case of the decision tree regressor, we use the mean squared
error, instead of the Gini metric, in order to determine how the tree is
built. You will learn about the mean squared error in detail in [Chapter
8](https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789343700/8),
*Performance Evaluation Methods*. In a nutshell, the mean squared error
is used to tell us about the prediction error rate.

Consider the tree shown in the following diagram:

![](./4_files/1e065149-da53-4321-9614-cbfe5e14c9ec.png)

An example decision tree for regression

When considering the preceding diagram of the decision tree, note the
following:

-   We are trying to predict the amount of a mobile transaction using
    the tree.
-   When the tree tries to decide on a split, it chooses the node in
    such a way that the target value is closest to the mean values of
    the target in that node.
-   You will notice that, as you go down the tree to the left, along the
    `True`{.literal} cases, the mean squared error of the nodes
    decreases.
-   Therefore, the nodes are built in a recursive fashion, such that it
    reduces the overall mean squared error, thereby obtaining the
    `True`{.literal} value.
-   In the preceding tree, if the old balance of origination is less
    than 600,281, then the amount (here, coded as `value`{.literal}) is
    80,442, and if it's greater than 600,281, then the amount is
    1,988,971.

 

 

#### Implementing the decision tree regressor in scikit-learn

In this section, you will learn how to implement the decision tree
regressor in scikit-learn. The first step is to import the data, and
create the features and target variables. We can do this using the
following code:

Copy

``` {.programlisting .language-markup}
import pandas as pd

#Reading in the dataset

df = pd.read_csv('fraud_prediction.csv')

#Dropping the index

df = df.drop(['Unnamed: 0'], axis = 1)

#Creating the features 

features = df.drop('amount', axis = 1).values
target = df['amount'].values
```

Note how, in the case of regression, the target variable is the amount,
and not the `isFraud`{.literal} column.

Next, we split the data into training and test sets, and build the
decision tree regressor, as shown in the following code:

Copy

``` {.programlisting .language-markup}
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#Splitting the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42)

#Building the decision tree regressor 

dt_reg = DecisionTreeRegressor(max_depth = 10, min_samples_leaf = 0.2, random_state= 50)

#Fitting the tree to the training data

dt_reg.fit(X_train, y_train)
```

 

 

In the preceding code, we do the following:

1.  We first import the required packages and split the data into
    training and test sets.
2.  Next, we build the decision tree regressor using the
    `DecisionTreeRegressor()`{.literal} function.
3.  We specify two hyperparameter arguments: `max_depth`{.literal},
    which tells the algorithm how many branches the tree must have, and
    `min_sample_leaf`{.literal}, which tells the tree about the minimum
    number of samples that each node must have. The latter is set to
    20%, or 0.2 of the total data, in this case.
4.  `random_state`{.literal} is set to 50 to ensure that the same tree
    is built every time we run the code.
5.  We then fit the tree to the training data.

#### Visualizing the decision tree regressor

Just as we visualized the decision tree classifier, we can also
visualize the decision tree regressor. Instead of showing you the
classes or categories to which the node of a tree belongs, you will now
be shown the value of the target variable.

We can visualize the decision tree regressor by using the following
code:

Copy

``` {.programlisting .language-markup}
#Package requirements 

from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import tree

#Extracting the feature names

feature_names = df.drop('amount', axis = 1)

#Creating the tree visualization

data = tree.export_graphviz(dt_reg, out_file=None, feature_names= feature_names.columns.values, proportion= True)

graph = pydotplus.graph_from_dot_data(data) 

# Show graph
Image(graph.create_png())
```

The code follows the exact same methodology as that of the decision tree
classifier, and will not be discussed in detail here. This produces a
decision tree regressor like that in the following diagram:

![](./4_files/16081b20-00db-4922-8d54-878a7df78749.png)

A visualization of the decision tree regressor

### The random forest regressor

The random forest regressor takes the decision tree regressor as the
base estimator, and makes predictions in a method similar to that of the
random forest classifier, as illustrated by the following diagram:

![](./4_files/382373f0-e107-446c-8dca-b177dc7a3e67.png)

Making the final prediction in the random forest regressor

The only difference between the random forest classifier and the random
forest regressor is the fact that, in the case of the latter, the base
estimator is a decision tree regressor.

#### Implementing the random forest regressor in scikit-learn

In this section, you will learn how you can implement the random forest
regressor in scikit-learn. The first step is to import the data and
split it into training and testing sets. This can be done using the
following code:

Copy

``` {.programlisting .language-markup}
import pandas as pd
from sklearn.model_selection import train_test_split

#Reading in the dataset

df = pd.read_csv('fraud_prediction.csv')

#Dropping the index

df = df.drop(['Unnamed: 0'], axis = 1)

#Creating the features and target arrays

features = df.drop('amount', axis = 1).values
target = df['amount'].values

#Splitting the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42)
```

The next step is to build the random forest regressor. We can do this
using the following code:

Copy

``` {.programlisting .language-markup}
from sklearn.ensemble import RandomForestRegressor

#Initiliazing an Random Forest Regressor with default parameters

rf_reg = RandomForestRegressor(max_depth = 10, min_samples_leaf = 0.2, random_state = 50)

#Fitting the regressor on the training data

rf_reg.fit(X_train, y_train)
```

 

 

In the preceding code, we do the following:

1.  We first import the `RandomForestRegressor`{.literal} module from
    scikit-learn.
2.  We then initialize a random forest regressor object, called
    `rf_reg`{.literal}, with a maximum depth of 10 for each decision
    tree, and the minimum number of data and samples in each tree as 20%
    of the total data.
3.  We then fit the tree to the training set.

### The gradient boosted tree

In this section, you will learn how the gradient boosted tree is used
for regression, and how you can implement this using scikit-learn.

In the AdaBoost classifier that you learned about earlier in this
chapter, weights are added to the examples that the classifier predicted
in correctly. In the gradient boosted tree, however, instead of weights,
the residual errors are used as labels in each tree in order to make
future predictions. This concept is illustrated for you in the following
diagram:

![](./4_files/2d9b8607-cf5d-45ca-8dfc-19f269a636d5.png)

Here is what occurs in the preceding diagram:

1.  The first decision tree is trained with the data that you have, and
    the target variable **Y**.
2.  We then compute the residual error for this tree.
3.  The residual error is given by the difference between the predicted
    value and the actual value.
4.  The second tree is now trained, using the residuals as the target.
5.  This process of building multiple trees is iterative, and continues
    for the number of base estimators that we have.
6.  The final prediction is made by adding the target value predicted by
    the first tree to the product of the shrinkage and the residuals for
    all the other trees.

7.  The shrinkage is a factor with which we control the rate at which we
    want this gradient boosting process to take place.
8.  A small value of shrinkage (learning rate) implies that the
    algorithm will learn more quickly, and therefore, must be
    compensated with a larger number of base estimators (that is,
    decision trees) in order to prevent overfitting.
9.  A larger value of shrinkage (learning rate) implies that the
    algorithm will learn more slowly, and thus requires fewer trees in
    order to reduce the computational time.

#### Implementing the gradient boosted tree in scikit-learn

In this section, we will learn how we can implement the gradient boosted
regressor in scikit-learn. The first step, as usual, is to import the
dataset, define the features and target arrays, and split the data into
training and test sets. This can be done using the following code:

Copy

``` {.programlisting .language-markup}
import pandas as pd
from sklearn.model_selection import train_test_split

#Reading in the dataset

df = pd.read_csv('fraud_prediction.csv')

#Dropping the index

df = df.drop(['Unnamed: 0'], axis = 1)

#Creating the features 

features = df.drop('amount', axis = 1).values
target = df['amount'].values

#Splitting the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42)
```

The next step is to build the gradient boosted regressor. This can be
done using the following code:

Copy

``` {.programlisting .language-markup}
from sklearn.ensemble import GradientBoostingRegressor

#Initializing an Gradient Boosted Regressor with default parameters

gb_reg = GradientBoostingRegressor(max_depth = 5, n_estimators = 100, learning_rate = 0.1, random_state = 50)

#Fitting the regressor on the training data

gb_reg.fit(X_train, y_train)
```

In the preceding code, we do the following:

1.  We first import `GradientBoostingRegressor`{.literal} from
    scikit-learn.
2.  We the build a gradient boosted regressor object with three main
    arguments: the maximum depth of each tree, the total number of
    trees, and the learning rate.
3.  We then fit the regressor on the training data.



Ensemble classifier
-------------------

* * * * *

The concept of ensemble learning was explored in this chapter, when we
learned about random forests, AdaBoost, andgradient boosted trees.
However, this concept can be extended to classifiers outside of trees.

If we had built a logistic regression, random forest, and k-nearest
neighbors classifiers, and we wanted to group them all together and
extract the final prediction through majority voting, then we could do
this by using the ensemble classifier.

This concept can be better understood with the aid of the following
diagram:

![](./5_files/b8e69d20-ec1f-495c-a8aa-40b9c448b0a5.png)

Ensemble learning with a voting classifier to predict fraud transactions

When examining the preceding diagram, note the following:

-   The random forest classifier predicted that a particular transaction
    was fraudulent, while the other two classifiers predicted that the
    transaction was not fraudulent.
-   The voting classifier sees that two out of three (that is, a
    majority) of the predictions are **Not Fraud**, and hence, outputs
    the final prediction as **Not Fraud**.

### Implementing the voting classifier in scikit-learn

In this section, you will learn how to implement the voting classifier
in scikit-learn. The first step is to import the data, create the
feature and target arrays, and create the training and testing splits.
This can be done using the following code:

Copy

``` {.programlisting .language-markup}
import pandas as pd
from sklearn.model_selection import train_test_split

#Reading in the dataset

df = pd.read_csv('fraud_prediction.csv')

#Dropping the index

df = df.drop(['Unnamed: 0'], axis = 1)

#Splitting the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42)
```

Next, we will build two classifiers that include the voting classifier:
the decision tree classifier and the random forest classifier. This can
be done using the following code:

Copy

``` {.programlisting .language-markup}
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Initializing the DT classifier

dt = DecisionTreeClassifier(criterion = 'gini', random_state = 50)

#Fitting on the training data

dt.fit(X_train, y_train)

#Initiliazing an Random Forest Classifier with default parameters

rf_classifier = RandomForestClassifier(random_state = 50)

#Fitting the classifier on the training data

rf_classifier.fit(X_train, y_train)
```

Next, we will build the voting classifier by using the following code:

Copy

``` {.programlisting .language-markup}
from sklearn.ensemble import VotingClassifier

#Creating a list of models

models = [('Decision Tree', dt), ('Random Forest', rf_classifier)]

#Initialize a voting classifier 

voting_model = VotingClassifier(estimators = models)

#Fitting the model to the training data

voting_model.fit(X_train, y_train)

#Evaluating the accuracy on the test data

voting_model.score(X_test, y_test)
```

In the preceding code, we do the following:

1.  We first import the `VotingClassifier`{.literal} module from
    scikit-learn.
2.  Next, we create a list of all the models that we want to use in our
    voting classifier.
3.  In the list of classifiers, each model is stored in a tuple, along
    with the model's name in a string and the model itself.
4.  We then initialize a voting classifier with the list of models built
    in step 2.
5.  Finally, the model is fitted to the training data and the accuracy
    is extracted from the test data.



Summary
-------

* * * * *

While this chapter was rather long, you have entered the world of tree
based algorithms, and left with a wide arsenal of tools that you can
implement in order to solve both small- and large-scale problems. To
summarize, you have learned the following:

-   How to use decision trees for classification and regression
-   How to use random forests for classification and regression
-   How to use AdaBoost for classification
-   How to use gradient boosted trees for regression
-   How the voting classifier can be used to build a single model out of
    different models

In the upcoming chapter, you will learn how you can work with data that
does not have a target variable or labels, and how to perform
unsupervised machine learning in order to solve such problems!