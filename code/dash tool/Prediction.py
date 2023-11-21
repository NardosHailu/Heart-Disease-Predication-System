from xml.etree.ElementPath import prepare_predicate
from dash import Dash, dcc, html, Input, Output, State
import numpy as np
import pickle

#load ML model
with open('prediction_model.pickle', 'rb') as f:
    clf = pickle.load(f)

#load ML model
with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

#set up
app = Dash(__name__)

#layout
app.layout = html.Div([
    html.Div([
        html.H1("Heart Disease Prediction System"),
        html.P("A system for predicting heart disease and its intensity"),
        
        html.Div([
            html.Div([
                html.Label("Patient information", className='Heading1'),

                html.Br(),
                html.Label('Age (years): ', className='Heading2'),
                html.Br(),
                dcc.Input(className='input', id='age', type='number', min=0, max=150, step=1),

                html.Br(),
                html.Label('Sex: ', className='Heading2'),
                dcc.Dropdown(className='dropdown', id='sex', 
                            options=[
                                {'label': 'Female', 'value': '0'},
                                {'label': 'Male', 'value': '1'},
                            ]),

                html.Br(),
                html.Label("Patient health", className='Heading1'),

                html.Br(),
                html.Label('Chest pain type: ', className='Heading2'),
                dcc.Dropdown(className='dropdown', id='cp', 
                            options=[
                                {'label': 'Typical angina', 'value': '1'},
                                {'label': 'Atypical angina', 'value': '2'},
                                {'label': 'Non-anginal pain', 'value': '3'},
                                {'label': 'Asymptomatic', 'value': '4'},
                                ]),
                    
                html.Br(),
                html.Label('Resting blood pressure (mmHg): ', className='Heading2'),
                html.Br(),
                dcc.Input(className='input', id='trestbps', type='number', min=0, max=200, step=1),
                    
                html.Br(),
                html.Label('Cholesterol (mg/dL): ', className='Heading2'),
                html.Br(),
                dcc.Input(className='input', id='chol', type='number', min=0, max=500, step=1),

                html.Br(),
                html.Label('High fasting blood sugar: ', className='Heading2'),
                dcc.Dropdown(className='dropdown', id='fbs', 
                            options=[
                                {'label': 'False', 'value': '0'},
                                {'label': 'True', 'value': '1'},
                                ]),

                html.Br(),
                html.Label('Exercise induced angina: ', className='Heading2'),
                dcc.Dropdown(className='dropdown', id='exang',
                            options=[
                                {'label': 'No', 'value': '0'},
                                {'label': 'Yes', 'value': '1'},
                                ]),

                html.Br(),
                html.Label('Maximum heart rate achieved (bpm): ', className='Heading2'),
                html.Br(),
                dcc.Input(className='input', id='thalach', type='number', min=0, max=200, step=1),
            ], style={'padding': 10, 'flex': 1}),

            html.Div([
                html.Label("ECG results", className='Heading1'),

                html.Br(),
                html.Label('Resting ECG: ', className='Heading2'),
                dcc.Dropdown(className='dropdown', id='restecg',
                            options=[
                                {'label': 'Normal', 'value': '0'},
                                {'label': 'ST-T wave abnormality', 'value': '1'},
                                {'label': 'Probable or definite left ventricular hypertrophy', 'value': '2'},
                                ]),

                html.Br(),
                html.Label('ST depression: ', className='Heading2'),
                html.Br(),
                dcc.Input(className='input', id='oldpeak'),

                html.Br(),
                html.Label('Peak ST slope: ', className='Heading2'),
                dcc.Dropdown(className='dropdown', id='slope',
                                options=[
                                    {'label': 'Upsloping', 'value': '1'},
                                    {'label': 'Flat', 'value': '2'},
                                    {'label': 'Downsloping', 'value': '3'},
                                    ]),

                html.Br(),
                html.Button(id='button', children="Predict", n_clicks=0)
            ], style={'padding': 10, 'flex': 1}),

        ], style={'display': 'flex', 'flex-direction': 'row'}),

        
    ], id='left-container'),

    html.Div([
        html.Br(),
        html.H1("Result", id='prediction', className='Heading3')
    ], id='right-container')
],id='container')

#callback for prediction
@app.callback(
    Output('prediction', 'children'),
    Input('button', 'n_clicks'),
    State('age', 'value'),
    State('sex', 'value'),
    State('cp', 'value'),
    State('trestbps', 'value'),
    State('chol', 'value'),
    State('fbs', 'value'),
    State('exang', 'value'),
    State('thalach', 'value'),
    State('restecg', 'value'),
    State('oldpeak', 'value'),
    State('slope', 'value')
)
def update_output(n_clicks, age, sex, cp, trestbps, chol, fbs, exang, thalach, restecg, oldpeak, slope):
#    x = np.array([[float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg),
#                float(thalach), float(exang), float(oldpeak), float(slope)]])
    if n_clicks == 0:
        return None


    x = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])
    for i in x[0]:
        if i is None:
            return 'incomplete input.'
    print(x)
    x = scaler.transform(x)
    print(x)
    prediction = clf.predict(x)[0]
    print(prediction)
    if prediction == 0:
        output = 'NO heart disease'
    elif prediction == 1:
        output = 'heart disease'
    return f'The prediciton result is {output}.'

#run the app
if __name__ == '__main__':
    app.run_server(debug=True)