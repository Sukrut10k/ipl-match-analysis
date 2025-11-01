# IPL Data Analysis with PySpark

A comprehensive data analytics project analyzing Indian Premier League (IPL) cricket matches using Apache Spark, featuring detailed business intelligence reports, interactive visualizations, and machine learning predictions.

## Project Overview

This project provides end-to-end analysis of IPL cricket data, including:
- **Match-level analytics** with detailed batting and bowling scorecards
- **Business Intelligence insights** with per-match and per-player metrics
- **Interactive visualizations** for performance trends and patterns
- **Machine learning models** for score and match outcome predictions

## Project Structure

```
IPL-Data-Analysis/
├── data/
│   ├── deliveries.csv          # Ball-by-ball delivery data
│   └── matches.csv              # Match-level information
├── IPL_data_analysis.ipynb      # Core analysis & scorecard generation
├── IPL_BI_Analysis.ipynb        # Business intelligence metrics
├── IPL_Visualizations.ipynb     # Interactive data visualizations
├── IPL_ML_Analysis.ipynb        # Machine learning predictions
└── README.md                    # Project documentation
```

## Technologies Used

- **Apache Spark 3.x** - Distributed data processing
- **PySpark** - Python API for Spark
- **Python 3.10+** - Core programming language
- **Pandas** - Data manipulation for ML workflows
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Machine learning algorithms
- **Jupyter Notebook** - Interactive development environment

## rerequisites

Before setting up the project, ensure you have:
- Python 3.10 or higher
- Java Development Kit (JDK) 8 or 11

## Detailed Local Setup Instructions

### Step 1: Install Java Development Kit (JDK)

Apache Spark requires Java to run. Install JDK 8 or 11:

**Windows:**
1. Download JDK from [Oracle](https://www.oracle.com/java/technologies/downloads/) or [Adoptium](https://adoptium.net/)
2. Run the installer and follow the installation wizard
3. Note the installation path (e.g., `C:\Program Files\Java\jdk-11.0.x`)

### Step 2: Install Apache Spark

**Windows:**
1. Download Spark from [Apache Spark Downloads](https://spark.apache.org/downloads.html)
   - Choose: Spark 3.5.0 (or latest)
   - Package type: Pre-built for Apache Hadoop 3.3
2. Extract the `.tgz` file to `C:\spark\` (you may need 7-Zip or WinRAR)
3. Your Spark installation should be at: `C:\spark\spark-3.5.0-bin-hadoop3`

### Step 3: Set Environment Variables

**Windows:**
1. Open "Environment Variables" settings:
   - Right-click "This PC" → Properties → Advanced System Settings → Environment Variables

2. Add **System Variables**:
   - Click "New" under System Variables
   
   | Variable Name | Value |
   |--------------|-------|
   | `JAVA_HOME` | `C:\Program Files\Java\jdk-11.0.x` |
   | `SPARK_HOME` | `C:\spark\spark-3.5.0-bin-hadoop3` |
   | `HADOOP_HOME` | `C:\spark\spark-3.5.0-bin-hadoop3` |
   | `PYSPARK_PYTHON` | `python` |

3. Edit **PATH** variable:
   - Select "Path" → Click "Edit"
   - Add these entries:
     ```
     %JAVA_HOME%\bin
     %SPARK_HOME%\bin
     %SPARK_HOME%\sbin
     ```

4. **Restart your computer** for changes to take effect

### Step 4: Verify Spark Installation

Test your Spark installation:

```bash
# Check Spark version
spark-submit --version

# Test PySpark shell (should open interactive Python shell with Spark)
pyspark

# In PySpark shell, test with:
# >>> spark.version
# >>> exit()
```

Expected output should show Spark version 3.x.x

### Step 5: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/IPL-Data-Analysis.git

# Navigate to project directory
cd IPL-Data-Analysis
```

### Step 6: Set Up Python Virtual Environment

**Windows:**
```bash
# Create virtual environment
python -m venv de_venv

# Activate virtual environment
de_venv\Scripts\activate
```

### Step 7: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install findspark pyspark pandas numpy scikit-learn plotly matplotlib seaborn jupyter
```

**Create requirements.txt** (for future reference):
```bash
pip freeze > requirements.txt
```

### Step 8: Prepare Data Files

1. Create a `data/` directory in the project root:
```bash
mkdir data
```

2. Download IPL datasets:
   - `deliveries.csv` - Ball-by-ball delivery data
   - `matches.csv` - Match information

3. Place both CSV files in the `data/` directory

**Data Sources:**
- [Kaggle IPL Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- Or use your own IPL datasets with similar schema

### Step 9: Configure Jupyter Notebook Kernel

```bash
# Install ipykernel
pip install ipykernel

# Add virtual environment as Jupyter kernel
python -m ipykernel install --user --name=de_venv --display-name="Python (DE venv)"
```

### Step 10: Launch Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

Your browser should open automatically. Navigate to the project directory and open any notebook.

### Step 11: Verify Setup

Open `IPL_data_analysis.ipynb` and run the first cell:

```python
import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("IPL Data Analysis") \
    .master("local[*]") \
    .getOrCreate()

print("Spark Session Created Successfully!")
print(f"Spark Version: {spark.version}")
```

If this runs without errors, your setup is complete!

---

## Notebook Descriptions

### 1. IPL_data_analysis.ipynb
**Core Match Analysis**
- Complete match scorecards (batting & bowling)
- Ball-by-ball delivery analysis
- Detailed commentary on PySpark transformations
- Schema optimization techniques
- IPL Final 2024 case study

**Features:**
- Automated scorecard generation
- Maiden over calculations
- Batting order determination
- Strike rate and economy calculations

### 2. IPL_BI_Analysis.ipynb
**Business Intelligence Insights**
- Per-match performance metrics
- Player-wise aggregated statistics
- Head-to-head batsman vs bowler analysis
- Match state tracking (run rate, wickets, etc.)
- Partnership analysis

**Metrics:**
- Innings runs, wickets, extras
- Runs per over
- Highest partnerships
- Batting averages and strike rates
- Bowling economy and dot ball percentage

### 3. IPL_Visualizations.ipynb
**Interactive Data Visualizations**
- Match timeline analysis with cumulative runs
- Top performers (batsmen & bowlers)
- Toss effect analysis
- Season-wise trend analysis
- Interactive Plotly charts

**Visualizations:**
- Line charts for run progression
- Bar charts for top performers
- Comparative analysis plots
- Trend analysis over seasons

### 4. IPL_ML_Analysis.ipynb
**Machine Learning Predictions**
- Final score prediction based on powerplay performance
- Match winner prediction using team form
- Feature engineering from match data
- Model evaluation with metrics

**ML Models:**
- Random Forest Regressor for score prediction
- Random Forest Classifier for winner prediction
- Feature importance analysis
- Model performance metrics (RMSE, Accuracy)

## 🔮 Future Scope

This project is actively evolving towards a comprehensive data engineering and analytics platform. Here's the detailed roadmap:

### Phase 1: Data Pipeline Development
**Goal: Build robust, automated ETL pipelines**

#### 1.1 Apache Airflow Integration
- **DAG Design**: Create modular, reusable Directed Acyclic Graphs (DAGs)
  - Daily match data ingestion DAG
  - Player statistics update DAG
  - Historical data backfill DAG
  - Data quality validation DAG
- **Task Orchestration**: 
  - Parallel processing of multiple data sources
  - Dynamic task generation based on match schedules
  - Retry logic and failure handling
  - Email/Slack notifications for pipeline failures
- **Scheduling**: 
  - Event-driven triggers for live match data
  - Scheduled batch processing for historical analysis
  - Incremental data loading strategies

#### 1.2 Advanced PySpark Transformations
- **Data Quality Framework**:
  - Schema validation and enforcement
  - Null value handling strategies
  - Outlier detection for match statistics
  - Data profiling and lineage tracking
- **Performance Optimization**:
  - Partition strategies for large datasets
  - Broadcast join optimization
  - Caching strategies for iterative queries
  - Adaptive Query Execution (AQE) tuning
- **Complex Transformations**:
  - Window functions for rolling averages
  - UDFs for custom cricket metrics (Duckworth-Lewis)
  - Pivot tables for multi-dimensional analysis
  - Graph analysis for player networks

### Phase 2: Data Warehouse Architecture
**Goal: Implement Medallion Architecture with Delta Lake**

#### 2.1 Bronze Layer (Raw Data Ingestion)
- **Data Sources**:
  - CSV/JSON match data files
  - Real-time API feeds (Cricinfo, ESPNCricinfo)
  - Web scraping for missing data
  - Social media sentiment data
- **Storage Strategy**:
  - Delta Lake for ACID transactions
  - Partitioning by season/match_date
  - Schema evolution support
  - Time travel capabilities for auditing

#### 2.2 Silver Layer (Cleaned & Validated Data)
- **Data Cleansing**:
  - Standardize player names across sources
  - Handle missing values with domain logic
  - Remove duplicate records
  - Normalize team names and venues
- **Data Enrichment**:
  - Add player age at match time
  - Calculate venue statistics
  - Weather data integration
  - Historical head-to-head records
- **Slowly Changing Dimensions (SCD)**:
  - Type 2 SCD for player profiles
  - Track team roster changes
  - Maintain venue information history

#### 2.3 Gold Layer (Business-Ready Analytics)
- **Aggregated Tables**:
  - Player career statistics
  - Team performance metrics
  - Venue-specific analysis
  - Season-wise summaries
- **Feature Store**:
  - Pre-computed ML features
  - Real-time feature serving
  - Feature versioning
  - Feature lineage tracking
- **Materialized Views**:
  - Top performers by season
  - Match outcome predictions
  - Player comparison metrics

### Phase 3: Advanced Analytics & BI
**Goal: Create interactive, real-time dashboards**

#### 3.1 Visualization Platform
- **Dashboard Technologies**:
  - **Power BI/Tableau**: Enterprise-grade reporting
  - **Apache Superset**: Open-source alternative
  - **Streamlit**: Custom Python dashboards
  - **Plotly Dash**: Interactive web applications
- **Dashboard Categories**:
  - **Executive Dashboards**: 
    - Season overview with KPIs
    - Team performance comparisons
    - Financial metrics (auction values)
  - **Coaching Dashboards**:
    - Player form analysis
    - Opposition analysis
    - Match strategy recommendations
  - **Fan Engagement Dashboards**:
    - Live match visualizations
    - Player comparisons
    - Fantasy team suggestions

#### 3.2 Business Intelligence Reports
- **Automated Reporting**:
  - Daily match reports
  - Weekly performance summaries
  - Monthly trend analysis
  - End-of-season comprehensive reports
- **Ad-hoc Analysis Tools**:
  - Self-service BI with natural language queries
  - Drill-down capabilities
  - Export to multiple formats (PDF, Excel)
- **Key Metrics & KPIs**:
  - Win probability models
  - Player impact scores
  - Team strength indicators
  - Venue advantage metrics

### Phase 4: Machine Learning Enhancement
**Goal: Deploy production-grade ML models**

#### 4.1 Advanced ML Models
- **Predictive Models**:
  - **Match Outcome Prediction**:
    - Gradient Boosting (XGBoost, LightGBM)
    - Neural networks for complex patterns
    - Ensemble methods for robustness
  - **Player Performance Forecasting**:
    - Time series models (ARIMA, Prophet)
    - Regression models for runs/wickets
    - Classification for form prediction
  - **Fantasy Points Prediction**:
    - Multi-output regression
    - Contextual models based on opposition
- **Deep Learning Applications**:
  - Shot classification from video analysis
  - Bowling action recognition
  - Injury prediction from workload data

#### 4.2 MLOps Implementation
- **Model Lifecycle Management**:
  - MLflow for experiment tracking
  - Model versioning and registry
  - A/B testing framework
  - Champion/challenger model deployment
- **Serving Infrastructure**:
  - Real-time inference API (FastAPI)
  - Batch prediction pipelines
  - Model monitoring and drift detection
  - Automated retraining triggers
- **Feature Engineering Pipeline**:
  - Automated feature extraction
  - Feature selection algorithms
  - Online feature computation
  - Feature store integration

### Phase 5: Real-Time Processing
**Goal: Enable live match analytics**

#### 5.1 Streaming Architecture
- **Technology Stack**:
  - Apache Kafka for event streaming
  - Spark Structured Streaming for processing
  - Redis for low-latency caching
  - WebSocket for real-time UI updates
- **Use Cases**:
  - Live win probability calculations
  - Real-time player performance tracking
  - Instant match commentary generation
  - Dynamic strategy recommendations

#### 5.2 Event-Driven Architecture
- **Event Sources**:
  - Ball-by-ball commentary feeds
  - Sensor data (ball speed, trajectory)
  - Video analytics events
  - Social media mentions
- **Processing Patterns**:
  - Complex event processing (CEP)
  - Stateful stream processing
  - Windowed aggregations
  - Late data handling

### Phase 6: Advanced Features
**Goal: Cutting-edge analytics capabilities**

#### 6.1 Natural Language Processing
- **Commentary Generation**:
  - Automated match summaries
  - Highlight reel descriptions
  - Player profile narratives
- **Sentiment Analysis**:
  - Fan reactions from social media
  - Player reputation scoring
  - Brand value estimation

#### 6.2 Computer Vision
- **Video Analysis**:
  - Shot detection and classification
  - Field placement optimization
  - Player movement tracking
  - Umpiring decision support

#### 6.3 Cloud Deployment
- **Platform Migration**:
  - AWS/Azure/GCP data lake setup
  - Kubernetes for container orchestration
  - Serverless functions for APIs
  - CI/CD pipelines with GitHub Actions
- **Scalability**:
  - Auto-scaling compute resources
  - Global content delivery networks
  - Multi-region redundancy

---
## Author
- LinkedIn: [Sukrut Kulkarni](https://www.linkedin.com/in/sukrut-kulkarni-b5a723290/)
- Email: k.sukrut1010@gmail.com
---

## Acknowledgments

- **Indian Premier League** for providing exciting cricket data
- **Apache Spark Community** for the powerful analytics framework
- **Kaggle Community** for IPL datasets
- **Cricket Data Analytics** researchers for methodologies

---