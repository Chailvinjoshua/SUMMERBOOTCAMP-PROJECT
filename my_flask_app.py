# my_flask_app.py
from flask import Flask, render_template, request, redirect, url_for, session, g, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import joblib
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor #ML IMPORT
from sklearn.tree import DecisionTreeRegressor #ML MODEL IMPORT
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'a_very_secret_and_complex_key_for_your_app_do_not_share_this'

# --- Database Setup ---
DATABASE = 'database.db'
SALES_DATA_CSV_PATH = 'sales_data.csv' # Path to the CSV for initial DB population

def get_db():
    """
    Establishes a connection to the SQLite database.
    The connection is stored in Flask's global context 'g' to be reused per request.
    """
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row # This allows accessing columns by name
    return db

@app.teardown_appcontext
def close_connection(exception):
    """
    Closes the database connection at the end of the request.
    """
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    """
    Initializes the database schema if the database file does not exist.
    Creates 'users' and 'sales_data' tables.
    Also populates 'sales_data' table from CSV if it's empty.
    """
    with app.app_context(): # Use app_context to access get_db outside of a request
        db = get_db()
        cursor = db.cursor()

        #A USER TABLES
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')

        #TABLE CREATE FOR SALES DATA
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sales_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL, -- Storing date as TEXT (YYYY-MM-DD)
                sales REAL NOT NULL
            )
        ''')
        db.commit()

        # POPULATE SALES DATA TABLE IF NOT EMPTY
        cursor.execute("SELECT COUNT(*) FROM sales_data")
        if cursor.fetchone()[0] == 0:
            print(f"Sales data table is empty. Attempting to load from {SALES_DATA_CSV_PATH}...")
            try:
                df_csv = pd.read_csv(SALES_DATA_CSV_PATH)
                df_csv['Date'] = pd.to_datetime(df_csv['Date']).dt.strftime('%Y-%m-%d') # Format for DB storage
                df_csv.to_sql('sales_data', db, if_exists='append', index=False)
                db.commit()
                print("Initial sales data loaded into database from CSV.")
            except FileNotFoundError:
                print(f"Warning: {SALES_DATA_CSV_PATH} not found. Sales data table remains empty.")
            except Exception as e:
                print(f"Error populating sales data table from CSV: {e}")
        else:
            print("Sales data table already contains data.")

    print("Database initialized or already exists.")


#MACHINE LEARNING TESTING AND TRAINING
MODEL_PATH = 'sales_forecast_model.pkl'
sales_model = None # Global variable to hold the loaded model

def load_sales_model():
    """
    Loads the trained ML model into the global sales_model variable.
    """
    global sales_model
    try:
        sales_model = joblib.load(MODEL_PATH)
        print(f"Machine learning model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
        sales_model = None # Ensure model is None if not found
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        sales_model = None #ENSURE MODEL NONE ERRORS

#TO LOAD MODEL PLEASE CALL THIS FUNCTION
load_sales_model()

def train_and_evaluate_model():
    """
    Trains multiple sales forecasting models (Linear Regression, RandomForest, DecisionTree)
    and returns their evaluation metrics. The best performing model (based on R2) is saved.
    """
    global sales_model
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT date, sales FROM sales_data ORDER BY date ASC")
    historical_records = cursor.fetchall()

    if not historical_records:
        return {"error": "No historical sales data found in the database to train the model.", "models": {}}

    df_train = pd.DataFrame(historical_records, columns=['Date', 'Sales'])
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    df_train['Days'] = (df_train['Date'] - df_train['Date'].min()).dt.days

    X = df_train[['Days']]
    y = df_train['Sales']

    if len(X) < 2: # ATLEAST 2 DATA POINTS.
        return {"error": "Not enough data points to train the model. Need at least 2.", "models": {}}

    #TRAINING & TESTING
    #ENSURE THE TRAINING AND TESTING
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else: # If only one data point, train on all of it
        X_train, y_train = X, y
        X_test, y_test = pd.DataFrame(), pd.Series() # Empty test sets

    model_results = {}
    best_r2 = -float('inf')
    best_model_name = ""
    best_model_instance = None

    models_to_train = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
    }

    for name, model_instance in models_to_train.items():
        try:
            model_instance.fit(X_train, y_train)
            rmse_val, r2_val = "N/A", "N/A"

            if not X_test.empty:
                y_pred = model_instance.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse_val = f"{np.sqrt(mse):.2f}"
                r2_val = f"{r2_score(y_test, y_pred):.2f}"

            model_results[name] = {
                "rmse": rmse_val,
                "r2": r2_val
            }

            # TRACK THE BEST MODEL RESULT FOR R2
            # Convert R2 to float for comparison, handle "N/A"
            current_r2 = -float('inf')
            if isinstance(r2_val, str) and r2_val != "N/A":
                try:
                    current_r2 = float(r2_val)
                except ValueError:
                    pass # Keep as -inf if not a valid number
            elif isinstance(r2_val, float):
                current_r2 = r2_val


            if current_r2 > best_r2:
                best_r2 = current_r2
                best_model_name = name
                best_model_instance = model_instance

        except Exception as e:
            model_results[name] = {"error": f"Training failed: {e}"}
            print(f"Error training {name}: {e}")

    # SAVE BEST MODEL FOR RESULTS ERROR
    if best_model_instance:
        joblib.dump(best_model_instance, MODEL_PATH)
        sales_model = best_model_instance # UPDATE THE MODEL INSTANCE
        print(f"Best model ({best_model_name}) saved as {MODEL_PATH}")
        model_results["best_model"] = best_model_name
    else:
        model_results["error"] = "No model could be trained successfully."
        print("No model could be trained successfully.")


    return {
        "success": True,
        "models": model_results
    }


def get_forecast_data_from_db():
    """
    Fetches historical sales data from the database and generates a forecast.
    Returns historical and forecasted data for charting.
    """
    if sales_model is None:
        print("Sales model not loaded. Cannot generate forecast.")
        return None

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT date, sales FROM sales_data ORDER BY date ASC")
    historical_records = cursor.fetchall()

    if not historical_records:
        print("No historical sales data found in the database.")
        return None

    # Convert fetched records to a pandas DataFrame for processing
    historical_df = pd.DataFrame(historical_records, columns=['Date', 'Sales'])
    historical_df['Date'] = pd.to_datetime(historical_df['Date'])
    historical_df['Days'] = (historical_df['Date'] - historical_df['Date'].min()).dt.days

    try:
        # TO DETERMINE THE LAST DATE OF THE DATA
        last_historical_date = historical_df['Date'].max()
        last_historical_day_index = historical_df['Days'].max()

        # NUMBER OF DAYS TO THE FORECAST
        forecast_period_days = 30  # FORECAST IN THE NEXT 30DAYS
    

        # Create future dates and their corresponding numerical 'Days' features
        future_days = np.array([last_historical_day_index + i for i in range(1, forecast_period_days + 1)]).reshape(-1, 1)
        future_dates = [last_historical_date + timedelta(days=i) for i in range(1, forecast_period_days + 1)]

        # Make predictions for future dates
        forecasted_sales = sales_model.predict(future_days)

        # Prepare historical data for JSON output
        historical_data_json = historical_df[['Date', 'Sales']].copy()
        historical_data_json['Date'] = historical_data_json['Date'].dt.strftime('%Y-%m-%d') # Format date for JSON

        # Prepare forecasted data for JSON output
        forecast_data_json = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
            'Sales': forecasted_sales
        })

        return {
            'historical': historical_data_json.to_dict(orient='records'),
            'forecast': forecast_data_json.to_dict(orient='records')
        }

    except Exception as e:
        print(f"An error occurred during forecasting: {e}")
        return None

# --- Flask Routes ---
@app.route('/')
def home():
    """
    Renders the login page as the default home page.
    """
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Handles user registration.
    GET: Displays the registration form.
    POST: Processes registration, hashes password, and stores user in DB.
    """
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        db = get_db()
        cursor = db.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            db.commit()
            flash('Registration successful! Please log in.', 'success') # Flash message
            return redirect(url_for('home'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error="Username already exists. Please choose a different one.")
        except Exception as e:
            return render_template('register.html', error=f"Registration failed: {e}")

    return render_template('register.html')

@app.route('/login', methods=['POST'])
def login():
    """
    Handles user login.
    Verifies credentials and sets session variable upon successful login.
    """
    username = request.form['username']
    password = request.form['password']

    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()

    if user and check_password_hash(user['password'], password):
        session['user'] = user['username']
        return redirect(url_for('sales_forecast_dashboard'))
    else:
        flash('Invalid username or password.', 'error') # Flash message
        return render_template('login.html', error="Invalid username or password.")

@app.route('/dashboard')
def sales_forecast_dashboard():
    """
    Displays the sales forecast dashboard.
    Requires user to be logged in.
    """
    if 'user' not in session:
        return redirect(url_for('home'))

    return render_template('sales_forecast.html', username=session['user'])

@app.route('/forecast_api')
def forecast_api():
    """
    API endpoint to provide historical and forecasted sales data for the chart.
    Data is sourced from the database.
    """
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = get_forecast_data_from_db()
    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "Could not generate forecast data. Model might not be trained or data is missing."}), 500

@app.route('/train_model_web')
def train_model_web():
    """
    Triggers model training and evaluation via a web request.
    Displays the training results for multiple models.
    """
    if 'user' not in session:
        flash('You must be logged in to train the model.', 'error')
        return redirect(url_for('home'))

    results = train_and_evaluate_model()

    # Ensure 'models' key is always present, even if training failed
    if "error" in results and "models" not in results:
        results["models"] = {} # Initialize empty dict if not present

    return render_template('model_training_results.html',
                           success=results.get('success', False),
                           message=results.get('error', ''),
                           models=results.get('models', {}),
                           best_model=results['models'].get('best_model', 'N/A') if results.get('models') else 'N/A',
                           username=session['user'])


@app.route('/logout')
def logout():
    """
    Logs out the user by removing their session data.
    """
    session.pop('user', None)
    return redirect(url_for('home'))

# --- Run the Application ---
if __name__ == '__main__':
    # Initialize the database when the application starts
    init_db()
    app.run(debug=True)
