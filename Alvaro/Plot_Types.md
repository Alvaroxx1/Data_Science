Data Science
### 01 - Basic Functions
---
### 02 - Numpy
* np.array - arange - zeros - linspace - eye - random.rand/randn/radint - arange
* reshape - max/argmax - min/argmin - dtype - array
* sqrt - exp - max - sin - log 
---
### 03 - Pandas
Usefull: describe() - mean() - max/min() - std() - count() - transpose() - info()
* Introduction to Pandas
* **Series**: Similar to an array but can have axis labels
* **DataFrames**: Bunch of Series objects put together to share the same index. pd.DtaFrame()
* **Missing Data**
    * df.dropna() - fillna()
* **GroupBy**
    * df.grpupby
* **Merging,Joining,and Concatenating**
    * concat() - merge(a,b,how='inner/outer/left/right',on=['key1'...]) - join() -> a.joing(b)
* **Operations**
    * df['col'].unique - nunique - value_counts - apply(function_name) - columns - index - df/sort_vaues() - isnull - dropna() - fillna - pivot_table - corr()
* **Data Input and Output**
    * df.read_csv - to_csv - read_excel('file',sheetname='') - to_excel - read_html - read_sql


---
### Pandas Build-in Data Visualization

* df.plot.area
* df.plot.barh
* df.plot.density
* df.plot.hist
* df.plot.line
* df.plot.scatter
* df.plot.bar
* df.plot.box
* df.plot.hexbin
* df.plot.kde
* df.plot.pie
---

### Using Cufflinks and iplot()

* scatter
* bar
* box
* spread
* ratio
* heatmap
* surface
* histogram
* bubble
---
### Choropleth Maps


### 11 Linear Regression Model

### 13 Logistic Regression
* Missing Data
* Cufflinks for plots

### 14 k Nearest Neighbors

* Import Libraries
* Get Data
* Standardize the variables
* Tran Test Split
* Using KNN
* Predictions and Evaluations
* Choosing a k value

### 15 Desicion Trees and Random Forest
* Import libraries
* Get the Data
* EDA
* Train Test Split
* Decision Trees
* Prediction and Evaluation
* Tree Visualization
* Random Forest

### 16 Support Vector Machines with Python
* Import Libraries
* Get the Data
* Set up DataFrame
* Train Test Split
* Train the Support Vector Classifier
* Gridserarch

### 17 - K means Clustering with Python
* Import libraries
* Create some Data
* Visualize Data
* Creating the Clusters

### 18 - Principal Component Analysis
* PCA Review
* Libraries
* The Data
* PCA visualization
* Interpreting the components

### 19 - Recomender system with python
* Import libraries
* Get the Data
* EDA
* Visualization Imports
* Recomeding Similar Movies

* __Advanced Recommender Systems with Python__
    * Using Content-Based and Collaborative Filtering (CF)
    * The Data
    * Getting Started
    * Train Test Split
    * Memory-Based Collaborative Filtering
    * Model Based Collaborative Filtering
    * SVD
### 20 - Natural Language Processing

* Get Data
* Exploratory Data Analysis
* Data Visualization
* Text Pre-processing
* Continuing Normalization
* Vectorization
* TF-IDF
* Training a model
* Model Evaluation
    * Precision & Recall
* Train Test Split
* Creating a Data Pipeline

### 21 - Big Data and Spark
1st
* Introduction to Spark and Python
* Creating a SparkContext
* Basic Operations
* Creating the RDD
* Actions
* Transformations
2nd
* RDD Transformations and Actions
    * RDD - Transformations - Actions - Spark Job
* Creating an RDD
* RDD Transformations
* RDD Actions
* Map vs flatMap
* RDD and Key Value Pairs
* Using Key Value Pairs for Operations

### 22 - Deep Learning

#### Tensor Flow Basics
* Simple Constants
* Running Sessions
* Operations
* Placeholder
* Defining Operations
#### MNIST Data-Set Basic Approach
* Get the MNIST Data
* Visualizing the Data
* Create the Model
* Create Session
#### Tensorflow with Estimators
* Get the Data
* Train Test Split
* Estimators
* Feature Columns
* Input Function
* Model Evaluation

### SciPy
Collection of mathematical algorithms and convenience functions built on the Numpy extension of Python
* Linear Algebra
    * linalg
* Sparse Linear Algebra
    * Linear Algebra for Sparse Matrices
