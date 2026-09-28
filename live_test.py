import os
import json
import sqlite3
import shutil
from src.bird_evaluator import run_bird_evaluation

def run_live_trace():
    test_dir = "live_trace_test"
    db_dir = os.path.join(test_dir, "dev_databases", "mock_db")
    os.makedirs(db_dir, exist_ok=True)
    
    # 1. Build a real SQLite database on disk
    db_path = os.path.join(db_dir, "mock_db.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Charlie', 35);")
    conn.commit()
    conn.close()

    # 2. Build a real BIRD-formatted dev.json
    dev_json_path = os.path.join(test_dir, "dev.json")
    with open(dev_json_path, 'w') as f:
        json.dump([{
            "question_id": 999,
            "db_id": "mock_db",
            "question": "What is the name of the oldest user?",
            "evidence": "Sort by age descending.",
            "SQL": "SELECT name FROM users ORDER BY age DESC LIMIT 1;"
        }], f)

    predict_path = os.path.join(test_dir, "predict.json")

    # 3. Execute the actual batch evaluation harness (Triggers real API calls)
    print("Sending live query to OpenRouter... This will take 5-15 seconds.")
    try:
        run_bird_evaluation(dev_json_path, os.path.join(test_dir, "dev_databases"), predict_path)
        print("✅ Success! The pipeline executed.")
        
        with open(predict_path, 'r') as f:
            print("\nOutput saved to predict.json:")
            print(json.dumps(json.load(f), indent=2))
            
    finally:
        # Clean up the 3-table temporary folder
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    run_live_trace()
