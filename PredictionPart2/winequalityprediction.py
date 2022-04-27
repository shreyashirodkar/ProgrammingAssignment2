import csv
import sys
import pandas
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType,DoubleType,IntegerType
from pyspark.sql.functions import isnull,when,count,col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def vectorAssemble(df,df_features):
    df_features.remove("quality")
    assembler = VectorAssembler(inputCols=df_features,outputCol="features")
    df_res = assembler.transform(df)
    return df_res

def evaluation(df):

    evaluator_f1 = MulticlassClassificationEvaluator(predictionCol="prediction",metricName="f1")
    f1_score = evaluator_f1.evaluate(df)
    print("fl score for model is: ",f1_score)

def dataprep():
    colNames=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
    
    schema = StructType().add("fixed acidity",DoubleType(),True).add("volatile acidity",DoubleType(),True).add("citric acid",DoubleType(),True).add("residual sugar",DoubleType(),True).add("chlorides",DoubleType(),True).add("free sulfur dioxide",IntegerType(),True).add("total sulfur dioxide", IntegerType(),True).add("density",DoubleType(),True).add("pH",DoubleType(),True).add("sulphates",DoubleType(),True).add("alcohol",DoubleType(),True).add("quality",IntegerType(),True)
    df = spark.read.format("csv").option("header",True).schema(schema).option("delimiter",";").load("/data/TrainingDataset.csv")
    df = df.replace('?',None).dropna(how='any')
    testdf = vectorAssemble(df,colNames).withColumnRenamed("quality","label")
    return testdf
    
if __name__ == "__main__":
   # if len(sys.argv) != 2:
    #    print("Usage: wordcount <file>", file=sys.stderr)
   #     sys.exit(-1)
    spark = SparkSession\
        .builder\
        .appName("PythonPA2")\
        .getOrCreate()
    
    print("Predicting wine quality...")
    test = dataprep()
    
    model = RandomForestClassificationModel.load("/model")
    df = model.transform(test)
    evaluation(df)
    
    spark.stop()