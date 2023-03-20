spark-submit --name pywordconut --verbose --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:log4j.properties" --executor-memory 2G --total-executor-cores 2 SimpleApp.py
