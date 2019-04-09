import numpy as np
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
import pyspark.sql.functions as F

from pyspark.mllib.stat import Statistics
from pyspark.ml.regression import GeneralizedLinearRegression, LinearRegression

import itertools


def prepare_data(df,labelCol,label_is_categorical,categoricalCols,continuousCols):
    
    feature_indexers = [StringIndexer(inputCol=col, 
                              outputCol="{0}_indexed".format(col))
                 for col in categoricalCols]

    feature_encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(),
                 outputCol="{0}_encoded".format(indexer.getOutputCol()))
                 for indexer in feature_indexers]

    feature_assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in feature_encoders]
                                + continuousCols, outputCol="features")

    pipeline = Pipeline(stages=feature_indexers + feature_encoders + [feature_assembler])

    if label_is_categorical is not None:  
        if label_is_categorical:
            label_indexer = StringIndexer(inputCol=labelCol, outputCol="label".format(labelCol))
            pipeline = Pipeline(stages=feature_indexers + feature_encoders + [feature_assembler] + [label_indexer])
        else:
            df = df.withColumn('label', F.col(labelCol))
        
    transformer = pipeline.fit(df)
    data = transformer.transform(df)

    return data


def estimate_correlation_matrix(df, cols, method='pearson', round_decimals=3):
    
    features = df.select(cols).rdd.map(lambda row: row[0:])
    corr_mat= pd.DataFrame(
        Statistics.corr(features, method=method), columns=cols, index=cols) \
        .round(round_decimals) \
        .style \
        .background_gradient(cmap='coolwarm')
    
    return corr_mat 

def predict_binary_category_based_on_threshold(prepared_df, model, threshold):
    
    predictions = model.transform(prepared_df)
    predictions = predictions.withColumn('predicted_category', 
                       F.when((F.col('prediction') > threshold), 1.0).otherwise(0.0))
    return predictions

def get_GLM_modelSummary(model):

    
    print ("Estimated model results:")
    print ("##","-------------------------------------------------")
    print ("##"," Estimate | Std.Error | t Values | P-value ")

    coef = np.append(list(model.coefficients),model.intercept)
    modelSummary=model.summary

    for i in range(len(modelSummary.pValues)):
        print ("##",
        '{:10.6f}'.format(coef[i]),\
        '{:10.4f}'.format(modelSummary.coefficientStandardErrors[i]),\
        '{:8.3f}'.format(modelSummary.tValues[i]),\
        '{:10.3f}'.format(modelSummary.pValues[i]))
    print('')    
    print("## Dispersion: {:3.2f}".format(modelSummary.dispersion))
    print("## Null Deviance: {:3.2f}".format(modelSummary.nullDeviance))
    print("## Residual Degree Of Freedom Null: {:3.2f}".format(modelSummary.residualDegreeOfFreedomNull))
    print("## Residual Degree Of Freedom: {:3.2f}".format(modelSummary.residualDegreeOfFreedom))
    print("## AIC: {:3.2f}".format(modelSummary.aic))

    
def get_linear_modelSummary(model):

    
    print ("Estimated model results:")
    print ("##","-------------------------------------------------")
    print ("##"," Estimate | Std.Error | t Values | P-value")
    coef = np.append(list(model.coefficients),model.intercept)
    Summary=model.summary

    for i in range(len(Summary.pValues)):
        print ("##",'{:10.3f}'.format(coef[i]),\
        '{:10.3f}'.format(Summary.coefficientStandardErrors[i]),\
        '{:8.3f}'.format(Summary.tValues[i]),\
        '{:10.3f}'.format(Summary.pValues[i]))

    print ("##",'---')
    print ("##","Mean squared error: {:.3f},".format(Summary.meanSquaredError), "RMSE: {:.3f}".format(Summary.rootMeanSquaredError)) 
    print ("##","Multiple R-squared: {:.3f},".format(Summary.r2), \
            "Total iterations:{:.0f}.".format(Summary.totalIterations))

def estimateVIF(df, Cols):

    categoricalCols, continuousCols, vif_values = [], [], []
    
    for col in list(Cols):
        data_type = str(df.schema[col].dataType)
        if data_type == 'StringType':
            categoricalCols.append(col)
        else:
            continuousCols.append(col)

    for col in Cols:

        data = prepare_data(df = df,
                            labelCol = col,
                            label_is_categorical = False,
                            categoricalCols = [Col for Col in categoricalCols if Col != col],
                            continuousCols = [Col for Col in continuousCols if Col != col]
                           )

        model = LinearRegression(featuresCol="features", labelCol='label')
        model_fit = model.fit(data)

        r2 = model_fit.summary.r2
        vif = 1/(1-r2) 

        vif_values.append(vif)
        result = dict(zip(Cols, vif_values))
        result = pd.DataFrame(
            sorted(result.items(), key=lambda x: x[1], reverse=True),
            columns=['Feature', 'VIF']).rename_axis('Rank')

    return result

def best_subset_selection_GLM(df, labelCol, Cols, label_is_categorical=False,family='gaussian', link='identity'):
    
    print('Total number of iterations: {}'.format(2**len(Cols)))
    
    AIC_values, feature_list, num_features = [], [], []

    for k in np.arange(1, len(Cols)+1):

        for i, combo in enumerate(itertools.combinations(Cols, k)):
                
            continuousCols, categoricalCols = [], []

            for col in list(combo):
                data_type = str(df.schema[col].dataType)
                if data_type == 'StringType':
                    categoricalCols.append(col)
                else:
                    continuousCols.append(col)

            data = prepare_data(df = df,
                                labelCol = labelCol,
                                label_is_categorical = False,
                                categoricalCols = categoricalCols,
                                continuousCols = continuousCols
                               )

            model = GeneralizedLinearRegression(family=family, 
                                              link=link, 
                                              featuresCol='features', 
                                              labelCol='label')

            AIC = model.fit(data).summary.aic
            AIC_values.append(AIC)

            feature_list.append(combo)
            num_features.append(len(combo))  
            
            print('Feature/s: {}, AIC={:.3f}'.format(combo, AIC))
      
    return pd.DataFrame({'num_features': num_features,'AIC': AIC_values, 'features':feature_list}).rename_axis('Model ID').sort_values('AIC', ascending=False)  