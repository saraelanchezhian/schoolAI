*****************************************************************************************
	      ReadMe.txt - Pre-requisites Installations - Package Details
*****************************************************************************************

-- Installations Required --
pip install numpy
pip install pandas
pip install matplotlib
pip install sklearn
pip install datetime
pip install pdfkit
pip install wkhtmltopdf

-- R package --
data_generator.R --> Generates data for training the network.
train_data_decision_tree.R --> Applies decision tree algorithm to assign threat level labels on train data
daily_data_decision_tree.R --> Applies decision tree algorithm to assign threat level labels on daily data
dotchart.R --> Generates a dotchart for the covariate probability distribution.

-- Python Package --
Main.py --> Main module that calls model to train the network and run predictions. Also, generates the final report.
classifier.py --> Module that defines the model for training and prediction.
split_test.py --> Module to split the input data into train and test chunks. Calls the model to train, test and generate the confusion matrix.


-- Input files --
school_shooter_db --> Main database containing simulated data for training the network.
daily_data --> Daily Data used for predicting threat levels

-- Dependent files --
Features.csv --> Required to run report.py
dotchart.csv --> Required to run dotchart in R

-- Misc files --
covariate_probability_distn.png --> probability distribution for the covariates.
 
-- Output files --
daily_output_data --> Daily Predicted data with threat levels assessed
report.html --> HTML version of the final output report
report.pdf --> PDF version of the final output report
piechart.png --> piechart of threat level assessed

