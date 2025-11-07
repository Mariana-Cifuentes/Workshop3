import os
import json
import joblib
import numpy as np
import pandas as pd
import mysql.connector
from kafka import KafkaConsumer

# Configuration

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_NAME = os.getenv("KAFKA_TOPIC", "happiness-topic")
MODEL_PATH = os.getenv("MODEL_PATH", "model/happiness_regression.pkl")
SPLIT_FILE = os.getenv("SPLIT_FILE", "artifacts/split_membership.csv") 

# MySQL connection
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASS = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DB   = os.getenv("MYSQL_DB", "etl_workshop3")

# Predictors (same order used during training)
PREDICTORS = [
    "GDP per Capita",
    "Healthy life expectancy",
    "Social support",
    "Freedom",
    "Generosity",
    "Perceptions of corruption",
]

# Database Helpers (Star Schema)

def open_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASS,
        database=MYSQL_DB
    )

def ensure_star_schema():
    """Create database and tables dim_country, dim_time, and fact_predictions if they do not exist."""
    cnn = mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASS
    )
    cnn.autocommit = True
    cur = cnn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB} CHARACTER SET utf8mb4;")
    cur.close()
    cnn.close()

    cnn = open_connection()
    cur = cnn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS dim_country (
          country_id INT AUTO_INCREMENT PRIMARY KEY,
          country_name VARCHAR(100) NOT NULL,
          UNIQUE KEY uk_country (country_name)
        ) ENGINE=InnoDB;
    """)
    cnn.commit()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS dim_time (
          time_id INT AUTO_INCREMENT PRIMARY KEY,
          year INT NOT NULL,
          UNIQUE KEY uk_year (year)
        ) ENGINE=InnoDB;
    """)
    cnn.commit()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS fact_predictions (
          fact_id BIGINT AUTO_INCREMENT PRIMARY KEY,
          country_id INT NOT NULL,
          time_id INT NOT NULL,

          gdp_per_capita DOUBLE,
          healthy_life_expectancy DOUBLE,
          social_support DOUBLE,
          freedom DOUBLE,
          generosity DOUBLE,
          perceptions_of_corruption DOUBLE,

          score_actual DOUBLE,
          score_pred DOUBLE,

          is_train TINYINT,
          is_test  TINYINT,

          CONSTRAINT fk_fact_country FOREIGN KEY (country_id)
            REFERENCES dim_country(country_id),
          CONSTRAINT fk_fact_time FOREIGN KEY (time_id)
            REFERENCES dim_time(time_id),

          KEY k_ct (country_id, time_id)
        ) ENGINE=InnoDB;
    """)
    cnn.commit()

    cur.close()
    cnn.close()

def get_or_create_country_id(cnn, country_name: str) -> int:
    cur = cnn.cursor()
    cur.execute("SELECT country_id FROM dim_country WHERE country_name=%s", (country_name,))
    row = cur.fetchone()
    if row:
        cur.close()
        return row[0]
    cur.execute("INSERT INTO dim_country (country_name) VALUES (%s)", (country_name,))
    cnn.commit()
    cid = cur.lastrowid
    cur.close()
    return cid

def get_or_create_time_id(cnn, year: int) -> int:
    cur = cnn.cursor()
    cur.execute("SELECT time_id FROM dim_time WHERE year=%s", (year,))
    row = cur.fetchone()
    if row:
        cur.close()
        return row[0]
    cur.execute("INSERT INTO dim_time (year) VALUES (%s)", (year,))
    cnn.commit()
    tid = cur.lastrowid
    cur.close()
    return tid

def insert_fact_prediction(cnn, record, y_pred, is_train, is_test):
  
    country = record.get("Country")
    year = record.get("Year")

    if country is None or year is None:
        return

    try:
        year = int(year)
    except:
        return

    country_id = get_or_create_country_id(cnn, country)
    time_id = get_or_create_time_id(cnn, year)

    def _to_float(x):
        if x is None:
            return None
        try:
            return float(x)
        except:
            return None

    vals = (
        country_id, time_id,
        _to_float(record.get("GDP per Capita")),
        _to_float(record.get("Healthy life expectancy")),
        _to_float(record.get("Social support")),
        _to_float(record.get("Freedom")),
        _to_float(record.get("Generosity")),
        _to_float(record.get("Perceptions of corruption")),
        _to_float(record.get("Score")),
        float(y_pred) if y_pred is not None else None,
        int(is_train) if is_train is not None else None,
        int(is_test)  if is_test  is not None else None,
    )

    cur = cnn.cursor()
    cur.execute(
        """
        INSERT INTO fact_predictions
        (country_id, time_id,
         gdp_per_capita, healthy_life_expectancy, social_support, freedom, generosity, perceptions_of_corruption,
         score_actual, score_pred, is_train, is_test)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        vals
    )
    cnn.commit()
    cur.close()


# Utilities

def make_key(country, year):
    if country is None or year is None:
        return None
    try:
        return f"{str(country).strip().lower()}|{int(year)}"
    except:
        return None

def main():
    
    ensure_star_schema()

    # Load pre-trained model 
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

    # Load split membership for is_train / is_test
    members = False
    train_keys, test_keys = set(), set()
    if os.path.exists(SPLIT_FILE):
        df_members = pd.read_csv(SPLIT_FILE)
        df_members['key'] = df_members['key'].astype(str)
        train_keys = set(df_members.loc[df_members['is_train'] == 1, 'key'])
        test_keys  = set(df_members.loc[df_members['is_test']  == 1, 'key'])
        members = True
        print(f"Split membership loaded: {SPLIT_FILE} | train={len(train_keys)} | test={len(test_keys)}")
    else:
        print("Warning: split_membership.csv not found. is_train/is_test will be stored as NULL.")

    # Initialize Kafka and MySQL
    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="happiness-consumer-group",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=300000  # stop after 5 minutes with no messages
    )
    cnn = open_connection()
    print("Kafka and MySQL ready (star schema). Waiting for messages...")

    processed = 0
    skipped = 0

    try:
        for msg in consumer:
            rec = msg.value

            # Validate that all predictors exist
            if any(rec.get(f) is None for f in PREDICTORS):
                skipped += 1
                continue

            # Feature vector (same order as training)
            try:
                x_vec = np.array([[float(rec[f]) for f in PREDICTORS]], dtype=float)
            except Exception:
                skipped += 1
                continue

            # Prediction
            try:
                y_pred = model.predict(x_vec)[0]
            except Exception:
                skipped += 1
                continue

            # Split flags
            key = make_key(rec.get("Country"), rec.get("Year"))
            if members and key is not None:
                is_train = 1 if key in train_keys else 0
                is_test  = 1 if key in test_keys  else 0
            else:
                is_train = None
                is_test  = None

            # Insert into fact table
            insert_fact_prediction(cnn, rec, y_pred, is_train, is_test)
            processed += 1

            if processed % 50 == 0:
                print(f"Processed: {processed} | Skipped: {skipped}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            cnn.close()
        except:
            pass
        consumer.close()
        print(f"Finished. Processed={processed}, Skipped={skipped}")

if __name__ == "__main__":
    main()