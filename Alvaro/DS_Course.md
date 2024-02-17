Data Science
### 01 - Basic Functions
---
### 02 - Numpy
* np.array - arange - zeros - linspace - eye - random.rand/randn/radint - arange
* reshape - max/argmax - min/argmin - dtype - array
* sqrt - exp - max - sin - log 
---
### 03 - Pandas
**Usefull**: describe() - mean() - max/min() - std() - count() - transpose() - info()
count: provide total number of rows
value_counts: tells the count for each and every category
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
### 04 - Pandas Exercises
1. Import Libraries
2. Read cvs/html/excel,etc.
3. Set it to a DataFrame
---
### 05 - Matplotlib Overview
**Introduction to Object Oriented Method**
* plt.plot() - x/ylabel() - title() - show() 
* subplot(r,c,plot_number)
* fig = plt.figure()
    * axes = fig.add_axes([left,bottom,width,height])
    * plot - set_x/ylabel - set_title
    
**Subplots**: act as an automatic axis manager
* plt.subplots(r,c)

**Figure size, aspect ration and DPI**
* fig = plt.figure(figsize,dpi)
    * figsize: tuple of the width and height
    * dpi: dots-per-inch

**Saving figures**
* fig.savefig('filename.png',dpi)

**Legends, labels and titles**
* ax = fig.add_axes([x,x,x,x])
```
ax.plot(x, x**2, label="x**2")
ax.plot(x, x**3, label="x**3")
ax.legend(loc)# loc 1,2,3,4
```

**Colors, linewidths,linetypes**
```
fig, ax = plt.subplots()
ax.plot(x, x**2, 'b.-') # g--
```
```
fig, ax = plt.subplots()

ax.plot(x, x+1, color="blue", alpha=0.5) # half-transparant
ax.plot(x, x+2, color="#8B008B")        # RGB hex code
ax.plot(x, x+3, color="#FF8C00")        # RGB hex code 
```
**Plot Range**
**Special Plot Types**
* scatter
* hist
* boxplot

**Advanced Matplotlib Concepts**
* Logarithmic Scale
* Placement of tics and sutom tick label
* Scientific notation
* Axis number and axis label spacing
* Axis position adjustments
* Axis grid
* Axis spines
* Twin axes
* Axes where x and y is zero
* 2D plot styles
    * Scatter - step - bar - fill_between
* Text Annotation
* Figure with multiple subplots and insets
* Subplot2grid
* gridspec
* add_axes
* Colormap and contour figures
    * pcolor
    * imshow
    * contour
* 3D figures
    * Surface plots
    * Wire-frame plot
    * Contour plots with projections
---

### 06 - Data visualization with Seaborn

**Distribution Plots**
* displot
* jointplot: match up two displots for bivariate data, compare with
    * scatter - reg - resid -kde - hex
* pairplot: pairwise relationship
* rugplot: dash mark for every point on a univariate distribution
* kdeplot: Kernel Density Estimation plots; observation with gaussian distribution centered around that variable

**Categorical Data Plots**
Boxplot and violin plot are used to shown the distribution of categorical data
The swarmplot is similar to stripplot(), but the points are adjusted so that they don’t overlap.
You can combine categorical plots
* factorplot
* boxplot: facilitate comparisons between variables 
* violinplot
* stripplot
* swarmplot
* barplot
* countplot: same as barplot but with only x value

**Matrix Plots**
Allow to plot data as color-encoded matrices, indicate clusters.
* Heatmap: in order for a heatmat to work properly, your data should already be in amatrix form 
* Clustermap: cluster version of a heatmap

**Grids**
Grid: type of plots that allow you to map plot types to rows and columns of a grid, similar plot; separate features
* PairGrid: pairwise relationships
* pairplot: simple version of pairGrid
* FacetGrid: grids of plots based off of a feature
* JoinGrid: general version of jointplot()

**Regression Plots**
* lmpltot: display linear models; split up those plots based off of features
* Working with Markers
* Using a Grid
* Aspect and Size

**Style and Color**
* Style
    * countplot ->set_style
* Spine Removal
    * despine
* Size and Aspect
* Scale and Context

**Examples**
* Import library
* Set style
* Load dataset
* Print the Head
* Display the plot as you like
    * Use numpy and seaborn
---

### 07 - Pandas Build-in Data Visualization
**Read Data with pandas**
* read_csv
    * Before plt.style() convert to hist -> 
    ``` 
    df1 = pd.read_csv('df1',index_col=0)
    df1['A'].hist()
    plt.style.use('bmh') #ggplot
    ``` 
**Plot Types**
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
**Example**
* Import libraries
* Display info() - head()
* Plot
---
### 08 - Using Cufflinks and iplot()

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
# Review Section 9
### 09 - Choropleth Maps 
Work with the graph offline
* Offline Plotly Usage
* Choropleth US Maps
---
### 10 - Working with Some Projects
**911 Calls Capstone Project**
* Import libraries (pandas, matplotlib, seaborn)
* Read dataset
* Info of the rows and columns
* Sort for top items
* Apply functions (lambda) and create new columns
* Display values (countplot)
* Separete the timestampt columns into diferent ones
* Use gruopby and plot(lplot,etc)
* Unstack method to change column and row order
* Display heatmap and clustermap
**Finance Project**
* Import (pandas,pandas_datareader,numpy,datetime)
* Download information of Banks within a time period
* Modify column names
* EDA (Exploratory Data Analysis)
    * pct_change(): percentage change in the values through a series
* Display (pairplot) and observe percentage change
    * pairplot - displot - heatmap - clustermap
* plot() - iplot()
* Analyze moving averages

### 11 Linear Regression Model
* Check out the data
* EDA: Exploratory Data Analysis
    * pairplot() - displot() - heatmap()
* Training Linear Regression Model
    1.Split data x(features) and y(target variable)
* Train Test Split
* Creating ans Training the Model
* Model Evaluation
* Predictions from our Model
* Regression Evaluation Metrics
    * **MAE**(Mean Absolute Error): average error
    * **MSE**(Mean Squared Error): punish larger errors
    * **RMSE**(Root Mean Squared Error): **RMSE**(Root Mean Squared Error) is interpretable in the 'y' units
Linear Regression Project
* Import libraries
* Get the Data
* 
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


**Key Terms**
* Standard Deviation
* percentage change


* **returns** refer to the income or profit generated by an investment,
