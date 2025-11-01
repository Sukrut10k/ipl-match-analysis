# IPL Data Analysis - Local PySpark Version
# Make sure deliveries.csv is in the same folder as this notebook

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import Window
import pyspark.sql.functions as F

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("IPL Data Analysis") \
    .master("local[*]") \
    .getOrCreate()

print("Spark Session Created Successfully!")

# ============================================
# 1. Load the CSV file
# ============================================
deliveries_df = spark.read.option('header', 'true').csv('deliveries.csv')
print("\n--- Initial Data Preview ---")
deliveries_df.show(5)

# ============================================
# 2. Check distinct match IDs
# ============================================
print("\n--- Distinct Match IDs (Sorted Descending) ---")
deliveries_df.select('match_id').distinct().sort(F.col("match_id").desc()).show()

# ============================================
# 3. Check initial schema
# ============================================
print("\n--- Initial Schema ---")
deliveries_df.printSchema()

# ============================================
# 4. Define proper schema with correct data types
# ============================================
int_col = ['match_id', 'inning', 'over', 'ball', 'batsman_runs',
           'extra_runs', 'total_runs', 'is_wicket']

fields = [StructField(col, IntegerType(), nullable=True) if col in int_col  
          else StructField(col, StringType(), nullable=True) for col in deliveries_df.columns]

# Reload with proper schema
deliveries_df = spark.read.option('header', 'true').schema(StructType(fields)).csv('deliveries.csv')
print("\n--- Updated Schema ---")
deliveries_df.printSchema()

# ============================================
# 5. Filter for IPL Final (match_id = 1426312)
# ============================================
ipl_final_df = deliveries_df.filter('match_id == 1426312')
print("\n--- IPL Final Match Data ---")
ipl_final_df.show(10)

# ============================================
# 6. First Innings Analysis
# ============================================
first_innings_batting = ipl_final_df.filter('inning == 1')
print("\n--- First Innings Data ---")
first_innings_batting.show(10)

# ============================================
# 7. BATTING SCORECARD
# ============================================
scorecard_df = first_innings_batting.filter("extras_type is NULL").groupBy('batter').agg(
    F.sum('batsman_runs').alias('runs'),
    F.count('ball').alias('balls'),
    F.count(F.when(first_innings_batting.batsman_runs == 4, 1)).alias('4s'),
    F.count(F.when(first_innings_batting.batsman_runs == 6, 1)).alias('6s'),
    F.round(F.sum('batsman_runs') * 100 / F.count('ball'), 2).alias('S/R')
)

# Get batting order
batsman_order = first_innings_batting.withColumn(
    'over-ball', 
    (F.concat(F.col("over"), F.lit("."), F.col("ball"))).cast(FloatType())
).groupBy("batter").agg(
    F.min("over-ball").alias("order")
).orderBy("order")

batting_order_df = batsman_order.withColumn(
    "batting_order", 
    F.row_number().over(Window.orderBy("order"))
)

# Join batting stats with batting order
batting_scorecard_final = scorecard_df.join(
    batting_order_df, 
    on=['batter'], 
    how='inner'
).select('batting_order', 'batter', 'runs', 'balls', '4s', '6s', 'S/R').orderBy('batting_order')

print("\n--- BATTING SCORECARD (First Innings) ---")
batting_scorecard_final.show()

# ============================================
# 8. BOWLING SCORECARD
# ============================================
scorecard_bowler_df = first_innings_batting.groupBy('bowler').agg(
    F.sum('total_runs').alias('runs_conceded'),
    F.sum((F.when((F.col("extras_type") == "legbyes") | (F.col("extras_type") == "byes"), 
                  F.col("extra_runs")))).alias('not_by_bowler'),
    F.count(F.when((F.col("extras_type").isNull()) | (F.col("extras_type") == "legbyes") | 
                   (F.col("extras_type") == "byes"), 1)).alias('balls'), 
    F.count(F.when(F.col("is_wicket") == 1, 1)).alias('W')
)

# Format bowling figures
scorecard_bowler_df = scorecard_bowler_df.select(
    F.col('bowler'),
    F.concat(F.floor(F.col('balls') / 6), F.lit("."), (F.col('balls') % 6)).alias('O'), 
    (F.col('runs_conceded') - F.coalesce(F.col('not_by_bowler'), F.lit(0))).alias('R'),
    F.col('W'),
    F.round((F.col('runs_conceded') / (F.col('balls') / 6)), 2).alias('Econ')
)

# ============================================
# 9. MAIDEN OVERS CALCULATION
# ============================================
maiden_bowler_df = first_innings_batting.groupBy('bowler', 'over').agg(
    F.sum('total_runs').alias('runs_conceded'),
    F.count(F.col('over')).alias('balls'),
    F.sum((F.when((F.col("extras_type") == "legbyes") | (F.col("extras_type") == "byes"), 
                  F.col("extra_runs")))).alias('not_by_bowler')
)

maiden_bowler_df = maiden_bowler_df.withColumn(
    'runs_by_bowler', 
    F.col('runs_conceded') - F.coalesce(F.col('not_by_bowler'), F.lit(0))
)

# Filter maiden overs (0 runs in 6 balls)
maiden_bowler_df = maiden_bowler_df.filter(
    (F.col('runs_by_bowler') == 0) & (F.col('balls') == 6)
).groupBy('bowler').agg(
    F.count('bowler').alias('M')
)

# Join bowling stats with maiden overs
bowling_scorecard_final = scorecard_bowler_df.join(
    maiden_bowler_df, 
    on=['bowler'], 
    how='left'
).fillna(value=0).select('bowler', 'O', 'M', 'R', 'W', 'Econ')

print("\n--- BOWLING SCORECARD (First Innings) ---")
bowling_scorecard_final.show()

# ============================================
# Optional: Save results to CSV
# ============================================
# batting_scorecard_final.coalesce(1).write.mode('overwrite').csv('batting_scorecard', header=True)
# bowling_scorecard_final.coalesce(1).write.mode('overwrite').csv('bowling_scorecard', header=True)

print("\n=== Analysis Complete ===")

# Stop Spark Session
# spark.stop()