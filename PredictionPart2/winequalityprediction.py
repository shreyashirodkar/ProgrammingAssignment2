import csv
import sys
import pandas
from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.feature import VectorAssembler


def vectorAssemble(df,df_features):
    df_features.remove('quality')
    assembler = VectorAssembler(inputCols=df_features,outputCol="features")
    df_res = assembler.transform(df)
    return df_res

def evaluation(df):

    evaluator_f1 = MulticlassClassificationEvaluator(predictionCol='prediction',metricName='f1')
    f1_score = evaluator_f1.evaluate(df)
    print("fl score for model is: ",f1_score)

def dataprep():
    colNames=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
    df = pandas.read_csv('/data/TrainingDataset.csv',sep=';',names=colNames,header=None,skiprows=1)
    testdf = vectorAssemble(df,colNames).withColumnRenamed('quality','label')
    return testdf
    
if __name__ == "__main__":
   # if len(sys.argv) != 2:
    #    print("Usage: wordcount <file>", file=sys.stderr)
   #     sys.exit(-1)
    spark = SparkSession\
        .builder\
        .appName("PythonPA2")\
        .getOrCreate()
    
    print("Hello World!")
    test = dataprep()
    
    model = RandomForestModel.load('/winemodel.model')
    df = model.transform(test)
    evaluation(df)
    
    spark.stop()