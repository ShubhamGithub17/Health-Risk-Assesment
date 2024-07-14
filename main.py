import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import base64
import tensorflow as tf
from keras.src.utils import img_to_array
from pygments.lexers import go
from sklearn.ensemble import RandomForestClassifier
from streamlit_option_menu import option_menu
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import plotly.express as px
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

heart_data = pd.read_csv("F:/DEGREE/SEM - 8/Project/Datasets for Para/diabetes_prediction_dataset.csv")

st.set_page_config(page_icon="💉", page_title="PARAMETER BASED HEALTH RISK ASSESSMENT")
st.markdown(
    """
        <style>
    [data-testid="stSidebarNavLink"]{
        visibility: hidden;
    }
    [data-testid="stSidebarUserContent"]{
        background: rgb(114,168,196);
        background: linear-gradient(90deg, rgba(114,168,196,1) 0%, rgba(200,148,235,0.9921218487394958) 100%);
    }
    section{
        background-color:#ccbad4;
    }
    [data-testid="stSidebarNavItems"]{
        background: rgb(114,168,196);
        background: linear-gradient(90deg, rgba(114,168,196,1) 0%, rgba(200,148,235,0.9921218487394958) 100%);
    }
    section{
        background: rgb(200,148,235);
        background: linear-gradient(90deg, rgba(200,148,235,0.9921218487394958) 0%, rgba(114,168,196,1) 100%);
    }
    [data-testid="stHeader"]{
        background: rgb(200,148,235);
        background: linear-gradient(90deg, rgba(200,148,235,0.9921218487394958) 14%, rgba(114,168,196,1) 100%);
    }
         .reportview-container {
            margin-top: -2em;
        }

        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        footer {visibility: hidden;}
    [data-testid="collapsedControl"] {
        display: none
    }
    </style>
    """,
    unsafe_allow_html=True
)

# main code
df = pd.read_csv("F:/DEGREE/SEM - 8/Project/Datasets for Para/diabetes_prediction_dataset.csv")
with st.sidebar:
    choose = option_menu("App Gallery", ["About", "Data Analysis (Diabetes)", "Data Visualization (Diabetes)",
                                         "Data Encoding (Diabetes)", "Diabetes Prediction", "Data Cleaning (Heart)",
                                         "Data Visualization (Heart)"
        , "Stroke Analysis", "Stroke Cleaning",
                                         "Stroke Visualization", "Life Expectancy", "Life Expectancy Visualization",
                                         "Bone Fracture", "Feedback", "LOGOUT"],
                         icons=['house', 'activity ', 'bar-chart-fill', 'bag-fill', 'person-lines-fill', 'recycle',
                                'bar-chart-line', 'activity', 'gear-wide', 'recycle', 'file-bar-graph-fill',
                                'heart-half', 'graph-up-arrow', 'person-rolodex', 'arrow-right-square-fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
                             "container": {"padding": "5!important", "background-color": "#fafafa"},
                             "icon": {"color": "orange", "font-size": "25px"},
                             "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                                          "--hover-color": "#eee"},
                             "nav-link-selected": {"background-color": "#02ab21"},
                         }
                         )


def about():
    st.title("About Health Assessment Project")
    st.write("""
        This project is designed to provide health assessment based on various parameters such as age, gender, hypertension, 
        heart disease, smoking history, BMI, HbA1c level, blood glucose level, and diabetes status.

        ### Features:
        - **Age:** Age of the individual.
        - **Gender:** Gender of the individual.
        - **Hypertension:** Whether the individual has hypertension (0: No, 1: Yes).
        - **Heart Disease:** Whether the individual has heart disease (0: No, 1: Yes).
        - **Smoking History:** Smoking history of the individual (never, former, current).
        - **BMI:** Body Mass Index of the individual.
        - **HbA1c Level:** HbA1c level of the individual.
        - **Blood Glucose Level:** Blood glucose level of the individual.
        - **Diabetes:** Diabetes status of the individual (0: No, 1: Yes).
        

        ### Tools Used:
        - **Streamlit:** For building the web application.
        - **Pandas:** For data manipulation.
        - **Seaborn and Matplotlib:** For data visualization.

        ### Contact:
        If you have any questions or suggestions, feel free to contact at Enroll no: 211260107525.
        """)


@st.cache_data
def Data_analysis():
    st.title("Data Analysis \n")
    st.write("-----")
    print()
    st.subheader("Data Overview:")
    st.write(df.head())
    st.write("-----")

    # Column Information
    st.subheader("Columns:")
    st.write(df.columns)
    st.write("---")

    # Missing Values
    st.subheader("Missing Values:")
    st.write(df.isnull().sum())
    st.write("---")

    # Shape of dataset
    st.subheader("Shape:")
    st.write(f"<span style='font-size:16pt; color:green'>{df.shape}</span>", unsafe_allow_html=True)
    st.write("----")
    # st.write("Info:\n")
    # st.write(df.info())

    # Description of HbA1c level
    st.subheader("What is HbA1c Level: \n")
    st.write(
        "HbA1c level, or glycated hemoglobin, is a measure of average blood glucose levels over the past 2-3 months. "
        "It is expressed as a percentage and is commonly used to diagnose and monitor diabetes.")

    # Find the highest and lowest HbA1c levels
    highest_hba1c = df["HbA1c_level"].max()
    lowest_hba1c = df["HbA1c_level"].min()

    st.write("**Highest HbA1c level:**", highest_hba1c)
    st.write("**Lowest HbA1c level:**", lowest_hba1c)

    # Find the highest and lowest blood glucose levels
    highest_glucose = df["blood_glucose_level"].max()
    lowest_glucose = df["blood_glucose_level"].min()
    st.write("**Highest blood glucose level:**", highest_glucose)
    st.write("**Lowest blood glucose level:**", lowest_glucose)

    # Set style
    sns.set_style("whitegrid")

    # Histogram for HbA1c levels
    fig_hba1c, ax_hba1c = plt.subplots()
    sns.histplot(df["HbA1c_level"], bins=20, kde=True, color='skyblue', edgecolor='black')
    ax_hba1c.set_xlabel('HbA1c Level', fontsize=12)
    ax_hba1c.set_ylabel('Frequency', fontsize=12)
    ax_hba1c.set_title('Distribution of HbA1c Levels', fontsize=14)
    plt.axvline(df["HbA1c_level"].mean(), color='red', linestyle='dashed', linewidth=1)  # Add mean line
    plt.legend(['Mean'])
    plt.tight_layout()
    st.pyplot(fig_hba1c)
    st.write("---")

    # Histogram for blood glucose levels
    fig_glucose, ax_glucose = plt.subplots()
    sns.histplot(df["blood_glucose_level"], bins=20, kde=True, color='lightgreen', edgecolor='black')
    ax_glucose.set_xlabel('Blood Glucose Level', fontsize=12)
    ax_glucose.set_ylabel('Frequency', fontsize=12)
    ax_glucose.set_title('Distribution of Blood Glucose Levels', fontsize=14)
    plt.axvline(df["blood_glucose_level"].mean(), color='red', linestyle='dashed', linewidth=1)  # Add mean line
    plt.legend(['Mean'])
    plt.tight_layout()
    st.pyplot(fig_glucose)

    """ Think of a histogram like a bar graph.In both histograms, Each bar shows how many times a certain blood sugar 
    or HbA1c level appears in our data. Taller bars mean more people have that level. So, by looking at the 
    histogram, we quickly see where most people's levels are.."""


@st.cache_data
def region():
    # Data Visualization
    st.title("DATA VISUALIZATION")
    # Histogram of age
    st.write("Histogram of Age:")
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.histplot(data=df, x='age', bins=20, kde=True, color='skyblue', edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Histogram of Age')
    st.write(fig)
    st.write("---")

    # Box plot of BMI by gender
    st.write("Box plot of BMI by gender:")
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(data=df, x='gender', y='bmi', palette='Set3', showfliers=True, linewidth=1.5, notch=True, whis=1.5)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('BMI', fontsize=12)
    plt.title('Box plot of BMI by Gender', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add median annotations
    medians = df.groupby('gender')['bmi'].median()
    for i, median in enumerate(medians):
        plt.text(i, median, f'Median: {median:.2f}', horizontalalignment='center', verticalalignment='bottom',
                 fontsize=10, color='red')

    # Customize whisker caps
    for whisker in ax.artists:
        whisker.set_linestyle('--')
        whisker.set_linewidth(1.5)

    st.pyplot(fig)
    st.write("---")

    # Count plot of hypertension by smoking history
    df['hypertension'] = df['hypertension'].map({0: 'No', 1: 'Yes'})
    st.write("Count plot of hypertension by smoking history:")
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.countplot(data=df, x='smoking_history', hue='hypertension', order=df['smoking_history'].value_counts().index)
    plt.xlabel('Smoking History')
    plt.ylabel('Count')
    plt.title('Count plot of Hypertension by Smoking History')
    # Add count values above each bar
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.1, height, ha='center', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    st.write(fig)
    st.write("---")

    # Pie chart of gender distribution
    st.write("Pie chart of Gender Distribution:")
    gender_distribution = df['gender'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(gender_distribution.values[:3], labels=gender_distribution.index[:3], autopct="%1.2f%%",
           explode=(0.1, 0.05, 0), shadow=True,
           textprops={'fontsize': 10, 'color': 'black', 'weight': 'bold'}, colors=("Lightpink", "Lightblue"))
    plt.title('Gender Distribution')
    st.write(fig)
    st.write("---")

    # Pie chart of smoking wise count
    st.write("Pie chart of Smoking Wise Count:")
    smoking_distribution = df['smoking_history'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))

    plt.pie(smoking_distribution.values[:6], labels=smoking_distribution.index[:6], autopct='%1.1f%%', startangle=140,
            explode=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05), shadow=True,
            textprops={'fontsize': 10, 'color': 'black', 'weight': 'bold'}
            )
    plt.title('Smoking Wise Count')
    st.write(fig)
    st.write("---")

    # Pie chart of diabetic and non-diabetic
    st.write("Pie chart of Diabetic and Non-Diabetic:")
    df['diabetes'] = df['diabetes'].map({0: 'Non-Diabetic', 1: 'Diabetic'})
    diabetes_distribution = df['diabetes'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))

    plt.pie(diabetes_distribution.values, labels=diabetes_distribution.index, autopct='%1.1f%%', startangle=140,
            shadow=True,
            explode=(0.2, 0), textprops={'fontsize': 10, 'color': 'black', 'weight': 'bold'},
            colors=("chocolate", "forestgreen"))
    plt.title('Diabetic and Non-Diabetic')
    st.write(fig)


def product():
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load the dataset
    @st.cache_resource
    def load_data():
        return pd.read_csv("F:/DEGREE/SEM - 8/Project/Datasets for Para/diabetes_prediction_dataset.csv")

    df = load_data()

    # Data preprocessing
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])

    ohe = OneHotEncoder()
    encoded_column = ohe.fit_transform(df[["smoking_history"]])
    df["smoking_history"] = encoded_column.toarray()

    # Split features and target variable
    x = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

    # Train the Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)

    # Streamlit UI
    st.title('Diabetes Risk Prediction')
    st.write('Enter the following details to predict diabetes risk:')

    pregnancies = st.number_input("**Pregnancy**", min_value=0, value=0)
    glucose = st.number_input("**Glucose Level** (Ranges from 70 mg/dL to 100 mg/dL)", min_value=0, value=0)
    blood_pressure = st.number_input("**Blood Pressure** (Ranges from less than 120 to 129)", min_value=0.0, value=0.0)
    skin_thickness = st.number_input("**Skin Thickness** (Ranges from 2.20-28.05 mm)", min_value=0.0, value=0.0)
    insulin = st.number_input("**Insulin Level** ( Ranges from 18 to 276 mIU/L)", min_value=0.0, value=0.0)
    bmi = st.number_input("""**BMI** \n  
    - Below 18.5 -> underweight
    - 18.5 - 24.9 -> Healthy
    - 25.0 - 29.9 -> overweight
    - 30 or over -> obese""", min_value=0.0, value=0.0)
    dpf = st.number_input("**Diabetes Pedigree Function** ( Ranges from 0.08 to 2.42)", min_value=0.0, value=0.0)
    age = st.number_input("**Age**", min_value=0, value=0)

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    # Make predictions
    if st.button('Predict'):
        prediction = rfc.predict(input_data)
        prediction_label = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        st.write(f"Predicted Diabetes Risk: {prediction_label}")

        # Model evaluation
        y_pred = rfc.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        st.write(f"Accuracy: {accuracy:.2f}%")


def contact():
    st.title("INFO")
    st.write("Please fill out the form below to get in touch with us.")

    # Input fields
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message")

    # Submit button
    if st.button("Submit"):
        if name and email and message:
            st.success("Thank you! Your message has been submitted.")

        else:
            st.error("Please fill out all fields.")


@st.cache_data
def encoding():
    st.subheader("DATA BEFORE ENCODING")
    st.dataframe(df)
    st.subheader("DATA AFTER ENCODING")
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])

    ohe = OneHotEncoder()
    encoded_column = ohe.fit_transform(df[["smoking_history"]])
    df["smoking_history"] = encoded_column.toarray()

    # Split features and target variable
    x = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    st.subheader("X TRAIN")
    st.dataframe(x_train)
    st.subheader("X TEST")
    st.dataframe(x_test)

    st.subheader("Y_TRAIN")
    st.dataframe(y_train)
    st.subheader("Y_TEST")
    st.dataframe(y_test)
    st.write("---")


@st.cache_data
def data_cleaning():
    st.subheader("HEART DATASET")
    st.dataframe(heart_data)
    st.write("----")

    st.subheader("ROWS")
    st.write(heart_data.shape[0])
    st.write("----")

    st.subheader("COLUMNS")
    st.write(heart_data.shape[1])
    st.write("----")

    st.subheader("NULL VALUES BY COLUMNS")
    st.dataframe(heart_data.isnull().sum())


def predictions():
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier

    # Load the dataset
    data = pd.read_csv("F:/DEGREE/SEM - 8/Project/Datasets for Para/framingham.csv")

    # Separate features (X) and target variable (y)
    X = data.drop('TenYearCHD', axis=1)

    # Initialize and train the XGBoost model
    model = XGBClassifier()
    model.fit(X, data['TenYearCHD'])

    # Define function to take user input and make predictions
    def predict_with_input():
        user_input = {}
        for feature in X.columns:
            user_input[feature] = st.number_input(feature, step=0.01)

        # Convert user input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Make prediction
        prediction = model.predict(input_df)

        if prediction[0] == 1:
            st.write("The model predicts that the individual has a 10-year risk of coronary heart disease (CHD).")
        else:
            st.write(
                "The model predicts that the individual does not have a 10-year risk of coronary heart disease (CHD).")

    # Streamlit UI
    st.title('CHD Risk Prediction')

    st.write("Enter the values for the following features:")
    predict_with_input()


@st.cache_data
def heart_visualization():
    # Load the data
    data = pd.read_csv("F:/DEGREE/SEM - 8/Project/Datasets for Para/framingham.csv")

    # Set the title
    st.title('Exploratory Data Analysis')

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 8))
    st.subheader('AGE Histogram')
    sns.histplot(data['age'], bins=20, alpha=0.7, kde=True, color='skyblue', edgecolor='black')
    st.write("\n ")
    plt.xlabel("Age", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    st.write("---")

    # Count plot for 'education' 0: Less than High School and High School degrees, 1: College Degree and Higher
    fig, ax = plt.subplots(figsize=(7, 6))
    st.subheader('Count Plot for Education')
    sns.countplot(x='education', data=data)
    st.write("\n ")
    ax.set_xlabel('Education', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    # Show percentages on top of bars
    total = float(len(data))
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.,
                height + 0.5,
                '{:.1%}'.format(height / total),
                ha="center")
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    st.write("---")

    # Count plot for 'currentSmoker'
    fig, ax = plt.subplots(figsize=(7, 6))
    st.subheader('Count Plot for Current Smoker')
    st.write("\n ")
    sns.countplot(x='currentSmoker', data=data, order=[0, 1])
    ax.set_xlabel('Current Smoker Status', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xticklabels(['Non-Smoker', 'Smoker'])
    st.pyplot(fig)
    st.write("---")

    # Count plot for 'BPMeds'
    data['BPMeds'] = data['BPMeds'].replace({0: "No", 1: "Yes"})
    fig, ax = plt.subplots(figsize=(10, 8))
    st.subheader('Count Plot for BPMeds')
    sns.countplot(x='BPMeds', data=data)
    ax.set_xlabel("BPMeds", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    st.pyplot(fig)
    st.write("---")

    # Count plot for 'prevalentStroke'
    data['prevalentStroke'] = data['prevalentStroke'].replace({0: "No", 1: "Yes"})
    fig, ax = plt.subplots(figsize=(8, 8))
    st.subheader('Count Plot for Prevalent Stroke')
    sns.countplot(x='prevalentStroke', data=data)
    ax.set_xlabel("Prevalent Stroke", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    st.pyplot(fig)
    st.write("---")

    # Count plot for 'prevalentHyp'
    data['prevalentHyp'] = data['prevalentHyp'].replace({0: "No", 1: "Yes"})
    fig, ax = plt.subplots(figsize=(8, 8))
    st.subheader('Count Plot for Prevalent Hyp')
    sns.countplot(x='prevalentHyp', data=data)
    ax.set_xlabel("Prevalent Hyp", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    st.pyplot(fig)
    st.write("---")

    # Count plot for 'diabetes'
    data['diabetes'] = data['diabetes'].replace({0: "No", 1: "Yes"})
    fig, ax = plt.subplots(figsize=(8, 8))
    st.subheader('Count Plot for Diabetes')
    sns.countplot(x='diabetes', data=data)
    ax.set_xlabel("Diabetes", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    st.pyplot(fig)
    st.write("---")


@st.cache_data
def stroke_analysis():
    s_df = pd.read_csv("F:/DEGREE/SEM - 8/Project/Datasets for Para/stroke_data.csv")

    # Basic Level Analysis
    st.title("Basic Data Analysis")

    st.write(s_df.head(5))
    # Average age
    st.subheader("1. Average Age:")
    average_age = s_df['age'].mean()
    st.write(f"The average age of individuals is {average_age:.2f} years.")

    # Count of males and females
    st.subheader("2. Count of Males and Females:")
    gender_counts = s_df['gender'].value_counts()
    st.write(gender_counts)

    # Percentage of individuals with hypertension
    st.subheader("3. Percentage of Individuals with Hypertension:")
    hypertension_percentage = (s_df['hypertension'].sum() / len(s_df)) * 100
    st.write(f"{hypertension_percentage:.2f}% of individuals have hypertension.")

    # Most common work type
    st.subheader("4. Most Common Work Type:")
    common_work_type = s_df['work_type'].mode()[0]
    st.write(f"The most common work type is {common_work_type}.")

    # Average BMI
    st.subheader("5. Average BMI:")
    average_bmi = s_df['bmi'].mean()
    st.write(f"The average BMI of individuals is {average_bmi:.2f}.")

    # Medium Level Analysis
    st.title("Medium Data Analysis")

    # Association between hypertension and heart disease
    st.subheader("6. Association between Hypertension and Heart Disease:")
    hypertension_heart_disease = pd.crosstab(s_df['hypertension'], s_df['heart_disease'])
    st.write(hypertension_heart_disease)

    # Distribution of smoking status among individuals with and without a history of stroke
    st.subheader("7. Distribution of Smoking Status among Individuals with and without Stroke:")
    smoking_stroke = pd.crosstab(s_df['stroke'], s_df['smoking_status'])
    st.write(smoking_stroke)

    # Difference in average glucose levels between smokers and non-smokers
    st.subheader("8. Difference in Average Glucose Levels between Smokers and Non-Smokers:")
    smokers_glucose = s_df[s_df['smoking_status'] == 'smokes']['avg_glucose_level'].mean()
    non_smokers_glucose = s_df[s_df['smoking_status'] == 'never smoked']['avg_glucose_level'].mean()
    st.write(f"Average glucose level of smokers: {smokers_glucose:.2f}")
    st.write(f"Average glucose level of non-smokers: {non_smokers_glucose:.2f}")

    # Patterns between work type and likelihood of stroke
    st.subheader("9. Patterns between Work Type and Likelihood of Stroke:")
    work_stroke = pd.crosstab(s_df['work_type'], s_df['stroke'])
    st.write(work_stroke)


@st.cache_data
def Stroke_Visualization():
    sv_df = pd.read_csv("F:/DEGREE/SEM - 8/Project/Datasets for Para/stroke_data.csv")

    # Medium Level Questions and Graphs
    st.title("Data Visualization")

    # 1. Correlation between age and average glucose level
    st.subheader("Correlation between Age and Average Glucose Level")

    import plotly.express as px
    from scipy.stats import linregress

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(sv_df['age'], sv_df['avg_glucose_level'])

    # Create scatter plot
    fig_correlation = px.scatter(sv_df, x='age', y='avg_glucose_level', color='gender',
                                 labels={'age': 'Age', 'avg_glucose_level': 'Average Glucose Level',
                                         'gender': 'Gender'})

    # Add trendline trace
    x_values = [min(sv_df['age']), max(sv_df['age'])]
    y_values = [slope * x + intercept for x in x_values]
    fig_correlation.add_trace(px.line(x=x_values, y=y_values).data[0])

    # Add reference line for normal glucose level range
    fig_correlation.add_shape(type="line", x0=min(sv_df['age']), x1=max(sv_df['age']), y0=100, y1=100,
                              line=dict(color="red", width=2, dash="dash"),
                              xref='x', yref='y')

    # Adjust axis ranges
    fig_correlation.update_layout(xaxis_range=[min(sv_df['age']) - 10, max(sv_df['age']) + 10],
                                  yaxis_range=[min(sv_df['avg_glucose_level']) - 10,
                                               max(sv_df['avg_glucose_level']) + 10])

    # Add tooltip
    fig_correlation.update_traces(hovertemplate='Age: %{x}<br>Avg Glucose Level: %{y}<br>Gender: %{marker.color}')

    # Add legend
    fig_correlation.update_layout(legend=dict(title='Gender'))

    # Add title and axis labels
    fig_correlation.update_layout(xaxis_title='Age',
                                  yaxis_title='Average Glucose Level')

    # Show plot
    st.plotly_chart(fig_correlation)
    st.write("---")

    # 2. Distribution of smoking status among individuals with and without a history of stroke
    st.subheader("Distribution of Smoking Status among Individuals with and without a History of Stroke")

    # Group data by stroke and smoking status
    smoking_stroke = sv_df.groupby(['stroke', 'smoking_status']).size().reset_index(name='count')

    # Calculate total counts for each stroke status
    total_counts = smoking_stroke.groupby('stroke')['count'].sum()

    # Pivot the DataFrame to get counts for each smoking status within each stroke status
    smoking_stroke_pivot = smoking_stroke.pivot_table(index='smoking_status', columns='stroke', values='count',
                                                      fill_value=0)

    # Calculate percentage of each smoking status within each stroke status
    smoking_stroke_pivot = smoking_stroke_pivot.div(total_counts, axis=1) * 100

    # Reset index to make 'smoking_status' a column again
    smoking_stroke_pivot.reset_index(inplace=True)

    # Melt the DataFrame to have 'stroke' and 'smoking_status' as columns
    smoking_stroke_melted = smoking_stroke_pivot.melt(id_vars='smoking_status', var_name='stroke',
                                                      value_name='percentage')

    # Map stroke values to labels
    stroke_labels = {0: 'No Stroke', 1: 'Stroke'}
    smoking_stroke_melted['stroke'] = smoking_stroke_melted['stroke'].map(stroke_labels)

    # Plot distribution of smoking status among individuals with and without stroke
    fig_smoking_stroke = px.bar(smoking_stroke_melted, x='smoking_status', y='percentage', color='stroke',
                                barmode='group',
                                labels={'percentage': 'Percentage', 'smoking_status': 'Smoking Status',
                                        'stroke': 'Stroke Status'})

    # Customize layout
    fig_smoking_stroke.update_layout(xaxis_title='Smoking Status',
                                     yaxis_title='Percentage',
                                     legend_title='Stroke Status',
                                     showlegend=True,
                                     bargap=0.1,
                                     plot_bgcolor='white',
                                     yaxis_gridcolor='lightgray')

    # Show plot
    st.plotly_chart(fig_smoking_stroke)
    st.write("---")

    # 3. Difference in average glucose levels between smokers and non-smokers
    st.subheader("Difference in Average Glucose Levels between Smokers and Non-Smokers")


    import plotly.graph_objects as go

    # Calculate average glucose levels for smokers and non-smokers
    smokers_glucose = sv_df[sv_df['smoking_status'] == 'smokes']['avg_glucose_level']
    non_smokers_glucose = sv_df[sv_df['smoking_status'] == 'never smoked']['avg_glucose_level']

    # Calculate mean and standard error for each group
    smokers_mean = smokers_glucose.mean()
    non_smokers_mean = non_smokers_glucose.mean()
    smokers_std = smokers_glucose.std()
    non_smokers_std = non_smokers_glucose.std()

    # Create bar plot with error bars
    fig_diff_glucose = go.Figure()
    fig_diff_glucose.add_trace(go.Bar(x=['Smokers', 'Non-Smokers'],
                                      y=[smokers_mean, non_smokers_mean],
                                      error_y=dict(type='data', array=[smokers_std, non_smokers_std],
                                                   visible=True), marker_color=['#1f77b4', '#ff7f0e']))

    # Add value on top of bars
    for i, mean in enumerate([smokers_mean, non_smokers_mean]):

        y_pos = mean + 0.1

        fig_diff_glucose.add_annotation(
            text=f"{mean:.2f}",
            x=['Smokers', 'Non-Smokers'][i],
            y=y_pos,
            showarrow=False,
            yanchor='bottom',
            xanchor='right',
        )

    # Customize layout
    fig_diff_glucose.update_layout(xaxis_title='Smoking Status',
                                   yaxis_title='Average Glucose Level',
                                   plot_bgcolor='rgba(0,0,0,0)',  # transparent background
                                   yaxis_gridcolor='lightgray',
                                   showlegend=False)

    # Show plot
    st.plotly_chart(fig_diff_glucose)
    st.write("---")

    # 4. Association between hypertension and heart disease
    sv_df['hypertension'] = sv_df['hypertension'].replace({0: "No", 1: "Yes"})
    sv_df['heart_disease'] = sv_df['heart_disease'].replace({0: "No", 1: "Yes"})
    st.subheader("Association between Hypertension and Heart Disease")
    hypertension_heart_disease = sv_df.groupby(['hypertension', 'heart_disease']).size().reset_index(name='count')
    fig_hypertension_heart_disease = px.bar(hypertension_heart_disease, x='hypertension', y='count',
                                            color='heart_disease', barmode='group',
                                            labels={'count': 'Count', 'hypertension': 'Hypertension',
                                                    'heart_disease': 'Heart Disease'})
    st.plotly_chart(fig_hypertension_heart_disease)
    st.write("---")

    # 5. Patterns between work type and likelihood of stroke
    st.subheader("Patterns between Work Type and Likelihood of Stroke")

    # Group data by work type and stroke status
    work_stroke = sv_df.groupby(['work_type', 'stroke']).size().reset_index(name='count')

    # Calculate total counts for each work type
    total_counts = work_stroke.groupby('work_type')['count'].sum()

    # Calculate percentage of stroke cases in each work type category
    work_stroke['percentage'] = (work_stroke['count'] / work_stroke.groupby('work_type')['count'].transform(
        'sum')) * 100

    # Round percentage values to two decimal places
    work_stroke['percentage'] = work_stroke['percentage'].round(2)

    # Plot patterns between work type and likelihood of stroke
    fig_work_stroke = px.bar(work_stroke, x='work_type', y='count', color='stroke', barmode='group',
                             labels={'count': 'Count', 'work_type': 'Work Type', 'stroke': 'Stroke Status'},
                             text='percentage',  # Display percentage as data labels
                             hover_data={'percentage': True},  # Show percentage in hover information
                             category_orders={'stroke': [0, 1]},  # Specify category order for legend
                             color_discrete_map={0: '#1f77b4', 1: '#ff7f0e'})  # Define color palette

    # Customize layout
    fig_work_stroke.update_layout(xaxis_title='Work Type',
                                  yaxis_title='Count',
                                  legend_title='Stroke Status',
                                  showlegend=True,
                                  bargap=0.1,
                                  plot_bgcolor='#f0f0f0',  # Light gray background
                                  yaxis_gridcolor='lightgray')

    # Show plot
    st.plotly_chart(fig_work_stroke)


def Stroke_Cleaning():
    import re
    sv_df = pd.read_csv("F:/DEGREE/SEM - 8/Project/Datasets for Para/stroke_data.csv")

    # Streamlit app
    st.title("Data Cleaning with Regular Expressions")

    # Display the original dataset
    st.subheader("Original Dataset:")
    st.write(sv_df)

    # Find total null values
    total_null = sv_df.isnull().sum().sum()
    st.write(f"Total Null Values: {total_null}")

    # Print number of null values for each column
    st.subheader("Null Values per Column:")
    null_per_column = sv_df.isnull().sum()
    st.write(null_per_column)

    # Remove rows with missing values
    if st.checkbox("Remove Rows with Missing Values"):
        sv_df_cleaned = sv_df.dropna()
        st.write("Rows with missing values have been removed.")
        st.write(sv_df_cleaned)

    # Fill null values in the 'bmi' column with mean or mode
    fill_method = st.radio("Select Fill Method:", ["Mean", "Mode"])

    if fill_method == "Mean":
        mean_bmi = sv_df['bmi'].mean()
        sv_df['bmi'] = sv_df['bmi'].fillna(mean_bmi)
        st.write("Null values in 'bmi' column have been filled with the mean value.")
    elif fill_method == "Mode":
        mode_bmi = sv_df['bmi'].mode()[0]
        sv_df['bmi'] = sv_df['bmi'].fillna(mode_bmi)
        st.write("Null values in 'bmi' column have been filled with the mode value.")

    # Display the cleaned dataset
    st.subheader("Cleaned Dataset:")
    st.write(sv_df)

    # Remove unnecessary characters from specific columns
    columns_to_clean = st.multiselect("Select Columns to Clean:", sv_df.columns)
    if columns_to_clean:
        for column in columns_to_clean:
            pattern = st.text_input(f"Enter regex pattern to remove from '{column}':")
            if pattern:
                sv_df[column] = sv_df[column].apply(lambda x: re.sub(pattern, '', str(x)))
                st.write(f"Regex pattern '{pattern}' has been removed from column '{column}'.")
        st.write(sv_df)

    column_to_remove = st.selectbox("Select Column:", sv_df.columns)
    value_to_remove = st.text_input(f"Enter Value to Remove from '{column_to_remove}':")

    # Remove rows with specified value
    if st.button("Remove Rows"):
        sv_df_cleaned = sv_df[sv_df[column_to_remove] != value_to_remove]
        st.write(f"Rows containing '{value_to_remove}' in column '{column_to_remove}' have been removed.")
        st.write("Remaining Data:")
        st.write(sv_df_cleaned)


def Life_Expectancy():
    def load_data():
        data = pd.read_csv("F:/DEGREE/SEM - 8/Project/Datasets for Para/Life Expectancy Data.csv")
        return data

    # Function to display null values and rows with null values
    def display_null_values(df):
        null_values = df.isnull().sum()
        st.write("Total Null Values in the Dataset:", null_values.sum())
        st.write("Null Values per Column:")
        st.write(null_values)

    # Function to fill null values
    def fill_null_values(df):
        # Fill null values with mean or mode depending on the field requirement
        df['Life expectancy '].fillna(df['Life expectancy '].mean(), inplace=True)
        df['Adult Mortality'].fillna(df['Adult Mortality'].mean(), inplace=True)
        df['Alcohol'].fillna(df['Alcohol'].mean(), inplace=True)
        df['Hepatitis B'].fillna(df['Hepatitis B'].mode()[0], inplace=True)
        df[' BMI '].fillna(df[' BMI '].mean(), inplace=True)
        df['Polio'].fillna(df['Polio'].mode()[0], inplace=True)
        df['Total expenditure'].fillna(df['Total expenditure'].mean(), inplace=True)
        df['Diphtheria '].fillna(df['Diphtheria '].mode()[0], inplace=True)
        df['GDP'].fillna(df['GDP'].mean(), inplace=True)
        df['Population'].fillna(df['Population'].mean(), inplace=True)
        df[' thinness  1-19 years'].fillna(df[' thinness  1-19 years'].mean(), inplace=True)
        df[' thinness 5-9 years'].fillna(df[' thinness 5-9 years'].mean(), inplace=True)
        df['Income composition of resources'].fillna(df['Income composition of resources'].mean(), inplace=True)
        df['Schooling'].fillna(df['Schooling'].mean(), inplace=True)

    # Main function
    def main():
        st.title("Life Expectancy Data Cleaning")

        # Load the data
        df = load_data()

        # Display null values
        st.subheader("Null Values in the Dataset Before Cleaning")
        display_null_values(df)

        # Fill null values
        fill_null_values(df)

        # Display null values again to confirm filling
        st.subheader("Null Values After Cleaning")
        display_null_values(df)

    # Run the main function
    if __name__ == "__main__":
        main()


def Life_Expectancy_Visualization():
    lev_df = pd.read_csv("F:/DEGREE/SEM - 8/Project/Datasets for Para/Life Expectancy Data.csv")

    # Sidebar
    st.title("Life Expectancy Data Visualization")
    visualization_option = st.selectbox(
        "Select Visualization",
        ["Select ", "Top and Bottom Countries by Life Expectancy",
         "GDP vs. Life Expectancy", "Alcohol Consumption vs. Life Expectancy",
         "BMI Distribution", "Prevalence of Thinness 5-9 Years", "Country-wise Life Expectancy"]
    )
    # Main content

    if visualization_option == "Top and Bottom Countries by Life Expectancy":
        top_countries = lev_df.groupby('Country')['Life expectancy '].mean().nlargest(10).index.tolist()
        bottom_countries = lev_df.groupby('Country')['Life expectancy '].mean().nsmallest(10).index.tolist()

        fig = px.bar(lev_df[lev_df['Country'].isin(top_countries + bottom_countries)],
                     x='Country', y='Life expectancy ',
                     title='Top and Bottom Countries by Life Expectancy',
                     color='Country')
        st.plotly_chart(fig)

    elif visualization_option == "GDP vs. Life Expectancy":
        fig = px.scatter(lev_df, x='GDP', y='Life expectancy ',
                         title='GDP vs. Life Expectancy',
                         trendline='ols',
                         labels={'GDP': 'GDP', 'Life expectancy ': 'Life Expectancy'})
        st.plotly_chart(fig)

    elif visualization_option == "Alcohol Consumption vs. Life Expectancy":
        fig = px.scatter(lev_df, x='Alcohol', y='Life expectancy ',
                         title='Alcohol Consumption vs. Life Expectancy',
                         trendline='ols',
                         labels={'Alcohol': 'Alcohol Consumption', 'Life expectancy ': 'Life Expectancy'})
        st.plotly_chart(fig)

    elif visualization_option == "BMI Distribution":
        fig = px.histogram(lev_df, x=' BMI ', title='BMI Distribution')
        st.plotly_chart(fig)

    elif visualization_option == "Prevalence of Thinness 5-9 Years":
        fig = px.bar(lev_df, x='Country', y=' thinness 5-9 years', title='Prevalence of Thinness 5-9 Years')
        st.plotly_chart(fig)

    elif visualization_option == "Country-wise Life Expectancy":
        avg_life_expectancy = lev_df.groupby('Country')['Life expectancy '].mean().reset_index()

        # Create choropleth map
        fig = px.choropleth(avg_life_expectancy,
                            locations='Country',
                            locationmode='country names',
                            color='Life expectancy ',
                            hover_name='Country',
                            title='Country-wise Life Expectancy',
                            color_continuous_scale='viridis',  # color scale
                            labels={'Life expectancy ': 'Life Expectancy'},  # Custom label
                            projection='natural earth',  # map projection
                            scope='world',  # map scope
                            template='plotly_dark',  # template
                            )

        # Customize the legend
        fig.update_layout(coloraxis_colorbar=dict(title='Life Expectancy', tickvals=[60, 70, 80, 90]))

        # Customize the layout
        fig.update_layout(
            width=800,
            height=600,
            geo=dict(
                showland=True,
                showocean=True,
                showlakes=True,
                showrivers=True,
                showcountries=True,
                landcolor='rgb(217, 217, 217)',
                oceancolor='rgb(103, 205, 255)',
                lakecolor='rgb(103, 205, 255)',
                rivercolor='rgb(103, 205, 255)',
                countrycolor='rgb(255, 255, 255)'
            )
        )

        # Add annotations
        fig.add_annotation(text='Source: Your Data Source', xref='paper', yref='paper', x=0.5, y=-0.1, showarrow=False)

        # Display the map
        st.plotly_chart(fig)


def fracture():
    model = tf.keras.models.load_model(
        'F:/DEGREE/SEM - 8/Project/Datasets for Para/xray_fracture_detection_model.h5')  # Load the model you saved

    # Define the classes
    classes = ['Fracture', 'Normal']

    def predict(image_path):
        print("Loading and preprocessing image...")
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        print("Image loaded and preprocessed. Shape:", img_array.shape)

        print("Making prediction...")
        prediction = model.predict(img_array)
        print("Prediction received. Shape:", prediction.shape)

        predicted_class = classes[int(np.round(prediction)[0][0])]
        return predicted_class

    # Streamlit app
    def main():
        st.title('X-ray Image Classification')
        st.write('Upload an X-ray image to classify it as "Fracture" or "Normal".')

        uploaded_file = st.file_uploader("Choose an X-ray image...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded X-ray Image.', use_column_width=True)

            # Make a prediction
            predicted_class = predict(uploaded_file)
            st.write('Prediction:', predicted_class)

    if __name__ == "__main__":
        main()


if choose == "About":
    about()
elif choose == "Data Analysis (Diabetes)":
    Data_analysis()
elif choose == "Data Visualization (Diabetes)":
    region()

elif choose == "Data Encoding (Diabetes)":
    encoding()
elif choose == "Diabetes Prediction":
    product()

elif choose == "Data Cleaning (Heart)":
    data_cleaning()
elif choose == "Data Visualization (Heart)":
    heart_visualization()

elif choose == "Stroke Analysis":
    stroke_analysis()
elif choose == "Stroke Cleaning":
    Stroke_Cleaning()
elif choose == "Stroke Visualization":
    Stroke_Visualization()

elif choose == "Life Expectancy":
    Life_Expectancy()
elif choose == "Life Expectancy Visualization":
    Life_Expectancy_Visualization()

elif choose == "Feedback":
    contact()

elif choose == "Bone Fracture":
    fracture()

elif choose == "LOGOUT":
    st.markdown(f'<meta http-equiv="refresh" content="2;url=http://localhost:8501/Login">', unsafe_allow_html=True)
    st.header("Redirecting...")
