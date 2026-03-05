import sys
import os
import shutil
import time
import traceback
from sklearn.ensemble import RandomForestClassifier as rf
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# inputs
training_data = 'data/titanic.csv'
include = ['Age', 'Sex', 'Embarked', 'Survived']
dependent_variable = include[-1]

model_directory = 'models'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory

# These will be populated at training time
model_columns = None
clf = None


@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', model_status=(clf is not None))


@app.route('/train-page')
def train_page():
    """Train page"""
    model_exists = os.path.exists(model_file_name)
    return render_template('train.html', model_exists=model_exists)


@app.route('/predict-page', methods=['GET', 'POST'])
def predict_page():
    """Prediction form and results page"""
    if request.method == 'POST':
        if not clf:
            return render_template('predict.html', model_status=False, error="Please train the model first")
        
        try:
            # Get form data
            age = float(request.form.get('Age'))
            sex = request.form.get('Sex')
            embarked = request.form.get('Embarked')
            
            # Create dataframe from form data
            data = [{
                'Age': age,
                'Sex': sex,
                'Embarked': embarked
            }]
            
            query = pd.get_dummies(pd.DataFrame(data))
            query = query.reindex(columns=model_columns, fill_value=0)
            
            prediction = clf.predict(query)[0]
            
            return render_template('predict.html', 
                                 model_status=True, 
                                 prediction=int(prediction))
            
        except Exception as e:
            return render_template('predict.html', 
                                 model_status=True, 
                                 error=str(e))
    
    return render_template('predict.html', model_status=(clf is not None))


@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))

            # https://github.com/amirziai/sklearnflask/issues/3
            # Thanks to @lorenzori
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(clf.predict(query))

            # Converting to int from int64
            return jsonify({"prediction": list(map(int, prediction))})

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'


@app.route('/train', methods=['GET'])
def train():
    # using random forest as an example
    # can do the training separately and just update the pickles

    df = pd.read_csv(training_data)
    df_ = df[include].copy()

    categoricals = []  # going to one-hot encode categorical variables

    for col, col_type in df_.dtypes.items():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df_[col] = df_[col].fillna(0)  # fill NA's with 0 for ints/floats

    # get_dummies effectively creates one-hot encoded variables
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(x.columns)
    
    # Create models directory if it doesn't exist
    os.makedirs(model_directory, exist_ok=True)
    
    joblib.dump(model_columns, model_columns_file_name)

    global clf
    clf = rf()
    start = time.time()
    clf.fit(x, y)

    joblib.dump(clf, model_file_name)

    message1 = 'Trained in %.5f seconds' % (time.time() - start)
    message2 = 'Model training score: %.2f%%' % (clf.score(x, y) * 100)
    return_message = 'Success! {0}. {1}.'.format(message1, message2)
    
    # Render template with success message
    return render_template('train.html', message=return_message, model_exists=True)


@app.route('/wipe', methods=['GET'])
def wipe():
    global clf, model_columns
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        clf = None
        model_columns = None
        return redirect(url_for('train_page'))

    except Exception as e:
        print(str(e))
        return render_template('train.html', error='Could not remove and recreate the model directory', model_exists=False)


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    try:
        clf = joblib.load(model_file_name)
        print('model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print('model columns loaded')

    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        clf = None

    app.run(host='0.0.0.0', port=port, debug=True)
