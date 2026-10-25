##############################################################

This module - The analytics pipeline shows two parts
Part A:
1.It shows profiling,cleaning and the datastory of data set.
2.part A code lives in "01_EDA.ipynb" file
3.Required libraries are loaded :
{
    pandas,matplotlib,seaborn,standardscaler
}
4."titanic.csv" dataset is loaded using seaborn and saved it locally also uploaded here.Also created the dataframe using pandas.
5.Profiling the dataset
   ->df.info()
   ->df.head()
   ->df.shape
   ->df.describe()
6.Computed the percentage of missing values in each column.
  ->"age" column is having missing values between 5%-30% so we need to impute it. since age column contains some outliers that i found out using the box plot uploaded here and also by computing IQR results.So imputed it by median because imputing median doesn't affect outliers.
  ->"deck" column is having 77% of missing values so we can drop the column. 
  ->"embarked" and "embark_town" column is having < 5% of missing values so we can drop those rows.since it is very less it doesn't affect the pattern of data.

Univariate analysis:
  ->plotted the box_plot and histogram for "age" and "fare" columns,the plots are uploaded here.
  ->computed IQR rule for outliers and reported the outliers of age and fare in the code.
  ->age values are right skewed i.e., mean>median>mode value "80" is an outlier.
  ->fare is right skewed and 512 is an outlier. mean>median>mode

Bivariate analysis:
 ->survival rate of males is 21.666666666666668
   survival rate of females is 73.70242214532871
   survival rate of females is more than males.
 ->survival rate of class 1 is 62.980769230769226
   survival rate of class 2 is 50.306748466257666
   survival rate of class 3 is 26.13065326633166
   survival rate of class 1 is more than all classes.
 ->survival rate of female in first class is 96.7032967032967
   survival rate of female in second class is 91.54929577464789
   survival rate of female in third class is 47.24409448818898
   survival rate of male in first class is 36.75213675213676
   survival rate of male in second class is 18.478260869565215
   survival rate of male in third class is 16.236162361623617
   Survival rate of females in class 1 is more.

Correlation matrix:
 ->top two highest abs(correlation) pairs - 
    (sibsp-parch)-0.38 weak correlation, 
    (pclass,fare)- 0.55 moderate correlation.


Multivariate analysis:
 ->plot 1 shows-females has higher survival rate than male, this is a histogram plot of survived and sex columns.
 ->first class passengers have higher survival rate followed by second and third classes.-this is a histogram for survived and pclass.
 ->Histogram of age distribution by survival-Younger age people have higher survival rates.
 ->scatter plot of survived of age vs fare-Passengers paying higher fares generally had better survival chances and young age people(especially children) have higher survival rate.

Exploratory check
 -> performed standardscaler jus for checking before after behaviour of mean and standard deviation of numerical columns.
 -> After scaling mean -0, standard deviation-1.

 Part B:
 this part code is inside the "02_modeling.ipynb" file
 imported libraries:
 {
    pandas
    time
    joblib
    numpy
    matplotlib
    seaborn
    train_test_split
    cross_val_score
    GridSearchCV 
    StratifiedKFold
    StandardScaler 
    OneHotEncoder
    ColumnTransformer
    Pipeline
    LogisticRegression,LinearRegression
    DecisionTreeClassifier,plot_tree
    RandomForestClassifier
    (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,mean_absolute_error,root_mean_squared_error,r2_score)
  }
  1.divided the columns into numeric and categoric columns.
  2.wrapped the preprocessing steps in ColumnTransfer().
  3.performed the train-test split of test size-0.2 using stratify method.
    ->stratification method is especially important for classification problems particularily when classes are imbalanced.
    -> stratification maintains the equal distribution of classes in train/test split.
4.later wrapped the preprocessor and models(LogisticRegression, DecisionTree, RandomForest) inside Pipeline(). 
 ->For titanic dataset recall is more important because it says of people who actually survived, how many did we correctly identify
 ->High recall means the model is missing fewer actual survivors. Also we should not completely ignore precision.
 ->Decision tree gives high recall.

Imbalance handling comparision for Decision tree
Decision Tree:
                 Accuracy  Precision   Recall
Baseline:         0.7662     0.8462     0.5238 
(class weight-    0.7273     0.67       0.65
  balanced)
SMOTE:            0.7273     0.67       0.65

  ->when class-weight is balanced the recall has increased but accuracy and precision decreased.
  -> class-weight balanced means to give more important to the minority class than majority class.so that bias will decrease.
  ->during smote oversampling(Synthetic minority over-sampling) there is not much difference because the data here is not highly imbalanced so smote is not changing anything.
  ->smote adds minority samples to the data by k nearest neighbours.

Hyperparameter tuning:
-> Gridsearch is applied on random forest with some parameters and best parameters and estimators are reported.
->Grid search runs with 27 Combinations X 5 Folds = 135 Fits
->Best parameters: {'model__max_depth': 5, 'model__min_samples_split': 2, 'model__n_estimators': 200}
->Best CV ROC-AUC: 0.8706 
->scores forthe best estimator of grid search
   ->Test Accuracy: 0.7922077922077922
   ->Test ROC_AUC score: 0.8236525379382522
   ->OOB Score:  0.8016260162601626

Linear Regression:
->linear regression appliedon the data set with target as "fare".and the residual plot is uploaded here.
->On observing the plot it seems like the residuals are randomly distributed around zero with approximately constant variance, the heteroscedasticity is not evident.
-> Also checked the heteroscedasticity statically with het_breuschpagan.

Final Observation:
As the data is a classification model and the data is not highly imbalanced I prefer using a classification model.
->On observing the results Logistic Regression with auc_roc score 8.3 is best and it also has significant better recall and precision than other models.

The pipeline is stored as pkl file using joblib.dump() and uploaded here.

##############################################################



