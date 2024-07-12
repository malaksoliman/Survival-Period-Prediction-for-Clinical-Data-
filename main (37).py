import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st

# Load the dataset
file_path = 'Cervix_Final_Fully_Encoded (1) (1).xlsx'
data = pd.read_excel(file_path)

# Define the dependent and independent variables
X = data.drop('Follow up Duration', axis=1)
y = data['Follow up Duration']

# Ensure only numeric data is used
X = X.select_dtypes(include=[np.number])
X.fillna(-1, inplace=True)  # Imputes missing values with -1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize machine learning models
models = {
    "Decision Tree": DecisionTreeRegressor(),
    "SVR": SVR(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(),
    "KNN": KNeighborsRegressor(),
    "ANN": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)  # Simple ANN with one hidden layer
}

# Train and evaluate each model
model_scores = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    model_scores[name] = (rmse, mae)
    print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Define the Streamlit app
st.title("Survival Period Prediction for Clinical Data")
st.write("## Welcome to the Survival Period Prediction Dashboard!")
st.write("### Please enter the patient data below:")

# User selects the model to use
model_choice = st.selectbox(
    "Choose the prediction model",
    ["Decision Tree", "SVR", "Random Forest", "Gradient Boosting", "KNN", "ANN"]
)

selected_model = models[model_choice]

# Define the input features and their options
categorical_features_options = {
    'Marital Status': ['Divorced', 'Married', 'Single', 'Widowed'],
    'Contraception': ['Depot medroxyprogesterone acetate', 'Intrauterine Device', 'No', 'Oral Contraceptive Pills'],
    'Pathology': ['Adenocarcinoma', 'Adenosquamous', 'Carcinosarcoma', 'Leiomyosarcoma', 'Squamous Cell Carcinoma', 'Undifferentiated Carcinoma'],
    'Concurrent Chemotherapy Protocol': ['Carboplatin', 'Cisplatin', 'Paclitaxel', 'Taxol'],
    'Induction Chemotherapy Protocol': ['5-fluorouracil /cisplatin', 'Paclitaxel/Carboplatin'],
    'Treatment types': ['Best Supportive Care', 'Chemotherapy', 'Surgery', 'Pelvic EBRT', 'Concurrent Chemotherapy', 'Brachytherapy', 'Induction Chemotherapy']
}

# Create a mapping of dropdown labels to feature columns
dropdown_features_mapping = {
    'Marital Status': ['Marital Status_Divorced', 'Marital Status_Married', 'Marital Status_Single', 'Marital Status_Widowed'],
    'Contraception': ['Contraception_Depot medroxyprogesterone acetate', 'Contraception_Intrauterine Device', 'Contraception_No', 'Contraception_Oral Contraceptive Pills'],
    'Pathology': ['Pathology_Adenocarcinoma', 'Pathology_Adenosquamous', 'Pathology_Carcinosarcoma', 'Pathology_Leiomyosarcoma', 'Pathology_Squamous Cell Carcinoma', 'Pathology_Undifferentiated Carcinoma'],
    'Concurrent Chemotherapy Protocol': ['Concurrent Chemotherapy Protocol_Carboplatin', 'Concurrent Chemotherapy Protocol_Cisplatin', 'Concurrent Chemotherapy Protocol_Paclitaxel', 'Concurrent Chemotherapy Protocol_Taxol'],
    'Induction Chemotherapy Protocol': ['Induction Chemotherapy Protocol_5-fluorouracil /cisplatin', 'Induction Chemotherapy Protocol_Paclitaxel/Carboplatin'],
    'Treatment types': ['Best Supportive Care', 'Chemotherapy', 'Surgery', 'Pelvic EBRT', 'Concurrent Chemotherapy', 'Brachytherapy', 'Induction Chemotherapy']
}

# Initialize a dictionary to hold user input
input_data_dict = {feature: 0 for feature in X.columns}

# Create number inputs
st.write("### Demographic Information")
cols1 = st.columns(2)
for i, feature in enumerate(['Age ', 'Age menarche', 'Age menopause', 'Age marriage', 'ECOG ', 'FIGO All', 'Grade All']):
    with cols1[i % 2]:
        if feature in ['ECOG ']:
            input_data_dict[feature] = st.slider(f"{feature}", min_value=0, max_value=10, value=0, key=f"{feature}_{i}")
        elif feature == 'FIGO All':
            input_data_dict[feature] = st.slider(f"{feature}", min_value=0, max_value=9, value=0, key=f"{feature}_{i}")
        elif feature == 'Grade All':
            input_data_dict[feature] = st.slider(f"{feature}", min_value=0, max_value=3, value=0, key=f"{feature}_{i}")
        else:
            input_data_dict[feature] = st.number_input(f"Enter {feature}", value=0, key=f"{feature}_{i}")

# Create yes/no inputs
st.write("### Medical History")
cols2 = st.columns(2)
for i, feature in enumerate(['DM', 'HTN', 'Cardiac', 'Hepatic', 'Renal', 'Abnormal Vaginal Bleeding', 'Abnormal Vaginal Discharge', 'Pelvic Pain', 'Constipation', 'Family History of Cervical Cancer']):
    with cols2[i % 2]:
        value = st.radio(f"{feature}", ('No', 'Yes'), key=f"{feature}_{i}")
        input_data_dict[feature] = 1 if value == 'Yes' else 0

# Create dropdown inputs
st.write("### Clinical Information")
cols3 = st.columns(2)
for feature_group, options in categorical_features_options.items():
    with cols3[list(categorical_features_options.keys()).index(feature_group) % 2]:
        value = st.selectbox(f"Select {feature_group}", options=options, key=feature_group)
        # Encode the selected option
        for option in options:
            column_name = dropdown_features_mapping[feature_group][options.index(option)]
            input_data_dict[column_name] = 1 if value == option else 0

# Convert input data dictionary to a DataFrame and ensure all columns are present
input_data = pd.DataFrame(input_data_dict, index=[0])
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Visualize the input data dynamically
st.write("### Input Data Visualization")
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(input_data, annot=True, cmap='coolwarm', cbar=False, ax=ax)
st.pyplot(fig)

# Predict the survival period
if st.button("Predict", key="predict_button"):
    prediction = selected_model.predict(input_data_scaled)
    st.write(f"The predicted survival period is: {prediction[0]:.2f} months")



