#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
from operator import add

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import isnull,when,count,col
from pyspark.sql.types import StructType,DoubleType,IntegerType
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier,DecisionTreeClassifier,GBTClassifier
from pyspark.ml.feature import VectorAssembler,VectorSlicer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.regression import RandomForestRegressor


def get_data(schema,path):
    df_train = spark.read.format("csv").option("header",True).schema(schema).option("delimiter",";").load(path)
    df_train.groupby('quality').count().show()

    df_train.select([count(when(isnull(c),c)).alias(c) for c in df_train.columns]).show()

    df_train = df_train.replace('?',None).dropna(how='any')

    meta = {"ml_attr": {"name": "quality",
                    "type": "nominal",
                    "vals": [str(x) for x in range(1,11)]}}
    df_train.withColumn("quality", col("quality").alias("quality", metadata=meta))

    return df_train

def evaluation(predMod,model,f):

    evaluator_f1 = MulticlassClassificationEvaluator(predictionCol='prediction',metricName='f1')
    evaluator_acc = MulticlassClassificationEvaluator(predictionCol='prediction',metricName='accuracy')
    f1_score = evaluator_f1.evaluate(predMod)
    acc_score = evaluator_acc.evaluate(predMod)
    f.writelines(["\n ",model," f1: ",str(f1_score)])
    f.writelines(["\n ",model," acc: ",str(acc_score)])


def vectorAssemble(df):
    df_features = df.columns
    df_features.remove('quality')
    assembler = VectorAssembler(inputCols=df_features,outputCol="features")
    df_res = assembler.transform(df)
    return df_res


def get_baseline(df,f,dfPred):
    if dfPred is None:
        pd = df.toPandas()
        feature_list = list(pd.columns)
        baseline_preds = pd['quality'].mean()
        baseline_errors = abs(baseline_preds - pd['quality'])
        f.writelines(['\n Average baseline error: ', str(round(np.mean(baseline_errors), 2))," \n"])



if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("PythonPA2")\
        .getOrCreate()

    schema = StructType().add("fixed acidity",DoubleType(),True).add("volatile acidity",DoubleType(),True).add("citric acid",DoubleType(),True).add("residual sugar",DoubleType(),True).add("chlorides",DoubleType(),True).add("free sulfur dioxide",DoubleType(),True).add("total sulfur dioxide", DoubleType(),True).add("density",DoubleType(),True).add("pH",DoubleType(),True).add("sulphates",DoubleType(),True).add("alcohol",DoubleType(),True).add("quality",IntegerType(),True)

    df_train = get_data(schema,"s3a://shreyaprogrammingassignment2winedata/TrainingDataset.csv")
    df_test = get_data(schema, "s3a://shreyaprogrammingassignment2winedata/ValidationDataset.csv")

    
    #See the correlations between variables 
    df_train_corr = df_train.toPandas()
    f = open("demofile2.txt", "w")
    f.write(df_train_corr.corr().to_string())
    get_baseline(df_test,f,None)

    
    #Create feature vector
    df_train_assembled = vectorAssemble(df_train)
    train=df_train_assembled.withColumnRenamed('quality','label')
    validation = vectorAssemble(df_test).withColumnRenamed('quality','label')
    

    #Testing different models for accuracy and f1
    lf = LogisticRegression(maxIter=10, regParam=0.01)
    lf_model = lf.fit(train)
    predLR = lf_model.transform(validation)
    evaluation(predLR,'logistic regression',f)

    rf = RandomForestClassifier(numTrees = 25)
    rf_model = rf.fit(train)
    predRF = rf_model.transform(validation)
    evaluation(predRF,'random forest',f)
    
    dt = DecisionTreeClassifier(featuresCol = 'features', maxDepth = 3)
    dtModel = dt.fit(train)
    predDT = dtModel.transform(validation)
    evaluation(predDT, 'decision tree',f)

    #crossvalidation
    grid = ParamGridBuilder().addGrid(rf.maxDepth,[3, 5, 8]).addGrid(rf.numTrees,[40,50,80]).build()
    evaluator = MulticlassClassificationEvaluator(metricName='f1')
 
    #Feature selection on best model - RF
    importanceList = pd.Series(rf_model.featureImportances.values)
    sorted_imp = importanceList.sort_values(ascending= False)
    kept = list((sorted_imp[sorted_imp > 0.03]).index)
    
    vs= VectorSlicer(inputCol= "features", outputCol="sliced", indices=kept)

    rf_slice = RandomForestClassifier(featuresCol='sliced')
    cv_sliced = CrossValidator(estimator=rf_slice,estimatorParamMaps=grid,evaluator=evaluator,parallelism =4)
    rf_model_slice = cv_sliced.fit(vs.transform(train))    
    predRFslice = rf_model_slice.transform(vs.transform(validation))
    evaluation(predRFslice,'\n random forest after slicing',f)
    
    rf_model.save("s3a://shreyaprogrammingassignment2winedata/winemodel.model")


    f.close()
    spark.stop()
