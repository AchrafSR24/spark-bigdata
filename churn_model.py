from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Churn-RF").getOrCreate()

data = spark.read.csv("/app/Data4.csv", header=True, inferSchema=True)

data = data.withColumn("TotalCharges", col("TotalCharges").cast("double"))

data = data.dropna()

label_indexer = StringIndexer(inputCol="Churn", outputCol="label")
data = label_indexer.fit(data).transform(data)

categorical_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                    "PaperlessBilling", "PaymentMethod"]

for cat_col in categorical_cols:
    indexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_index")
    data = indexer.fit(data).transform(data)

feature_cols = [col + "_index" for col in categorical_cols] + ["tenure", "MonthlyCharges", "TotalCharges"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_data = assembler.transform(data).select("features", "label")

train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestClassifier(numTrees=50)
model = rf.fit(train_data)
results = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(results)
print(f" Accuracy du Random Forest : {accuracy:.2f}")

spark.stop()
