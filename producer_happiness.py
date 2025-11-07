import os
import json
import time
import numpy as np
import pandas as pd
from kafka import KafkaProducer


#CONFIGURATION

DATA_PATH = "data"
TOPIC_NAME = "happiness-topic"
BOOTSTRAP_SERVERS = "localhost:9092"
SLEEP_TIME = 0.01

# COLUMN RENAMING 

rename_dicts = {
    "2015": {
        "Country": "Country",
        "Region": "Region",
        "Happiness Rank": "Rank",
        "Happiness Score": "Score",
        "Economy (GDP per Capita)": "GDP per Capita",
        "Family": "Social support",
        "Health (Life Expectancy)": "Healthy life expectancy",
        "Freedom": "Freedom",
        "Trust (Government Corruption)": "Perceptions of corruption",
        "Generosity": "Generosity"
    },
    "2016": {
        "Country": "Country",
        "Region": "Region",
        "Happiness Rank": "Rank",
        "Happiness Score": "Score",
        "Economy (GDP per Capita)": "GDP per Capita",
        "Family": "Social support",
        "Health (Life Expectancy)": "Healthy life expectancy",
        "Freedom": "Freedom",
        "Trust (Government Corruption)": "Perceptions of corruption",
        "Generosity": "Generosity"
    },
    "2017": {
        "Country": "Country",
        "Happiness.Rank": "Rank",
        "Happiness.Score": "Score",
        "Economy..GDP.per.Capita.": "GDP per Capita",
        "Family": "Social support",
        "Health..Life.Expectancy.": "Healthy life expectancy",
        "Freedom": "Freedom",
        "Generosity": "Generosity",
        "Trust..Government.Corruption.": "Perceptions of corruption"
    },
    "2018": {
        "Country or region": "Country",
        "Overall rank": "Rank",
        "Score": "Score",
        "GDP per capita": "GDP per Capita",
        "Social support": "Social support",
        "Healthy life expectancy": "Healthy life expectancy",
        "Freedom to make life choices": "Freedom",
        "Generosity": "Generosity",
        "Perceptions of corruption": "Perceptions of corruption"
    },
    "2019": {
        "Country or region": "Country",
        "Overall rank": "Rank",
        "Score": "Score",
        "GDP per capita": "GDP per Capita",
        "Social support": "Social support",
        "Healthy life expectancy": "Healthy life expectancy",
        "Freedom to make life choices": "Freedom",
        "Generosity": "Generosity",
        "Perceptions of corruption": "Perceptions of corruption"
    }
}

final_columns = [
    "Country", "Freedom", "GDP per Capita", "Generosity",
    "Healthy life expectancy", "Perceptions of corruption",
    "Score", "Social support", "Year"
]

# NORMALIZE COUNTRY NAMES

def normalize_country(name):
    name = str(name).strip().lower()
    replacements = {
        "hong kong s.a.r., china": "hong kong",
        "taiwan province of china": "taiwan",
        "somaliland region": "somaliland",
        "north macedonia": "macedonia",
        "swaziland": "eswatini",
        "trinidad and tobago": "trinidad & tobago",
        "north cyprus": "northern cyprus"
    }
    return replacements.get(name, name)


# LOAD AND CLEAN DATA

def load_and_transform():
    df_2015 = pd.read_csv(os.path.join(DATA_PATH, "2015.csv"))
    df_2016 = pd.read_csv(os.path.join(DATA_PATH, "2016.csv"))
    df_2017 = pd.read_csv(os.path.join(DATA_PATH, "2017.csv"))
    df_2018 = pd.read_csv(os.path.join(DATA_PATH, "2018.csv"))
    df_2019 = pd.read_csv(os.path.join(DATA_PATH, "2019.csv"))

    datasets = {
        "2015": df_2015, "2016": df_2016, "2017": df_2017, "2018": df_2018, "2019": df_2019
    }

    for year, df in datasets.items():
        df.rename(columns=rename_dicts[year], inplace=True)
        df["Year"] = int(year)

    df_all = pd.concat(
        [
            datasets["2015"][final_columns],
            datasets["2016"][final_columns],
            datasets["2017"][final_columns],
            datasets["2018"][final_columns],
            datasets["2019"][final_columns]
        ],
        ignore_index=True
    )

    df_all["Country"] = df_all["Country"].apply(normalize_country)
    df_all["Perceptions of corruption"] = df_all["Perceptions of corruption"].interpolate(method="linear")

    print(df_all.info())
    print(df_all.head())
    print("NaN values in 'Perceptions of corruption':", df_all["Perceptions of corruption"].isna().sum())

    return df_all

# SEND TO KAFKA

def main():
    print("Loading and cleaning datasets (strict EDA)...")
    df_all = load_and_transform()

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8")
    )

    print(f"Sending {len(df_all)} records to topic '{TOPIC_NAME}' ...")
    sent = 0
    try:
        for _, row in df_all.iterrows():
            record = row.to_dict()
            for k, v in record.items():
                if isinstance(v, float) and np.isnan(v):
                    record[k] = None
            producer.send(TOPIC_NAME, value=record)
            sent += 1
            time.sleep(SLEEP_TIME)
            if sent % 100 == 0:
                print(f"Sent {sent} records...")
        producer.flush()
        print(f"Producer completed. Total records sent: {sent}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        producer.close()
        print("Producer closed.")

# DIRECT EXECUTION

if __name__ == "__main__":
    main()