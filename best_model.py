import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile
import pickle
from xgboost import XGBRegressor
import plotly.express as px
hide_elements_css = """
<style>
/* Ẩn biểu tượng GitHub và các lớp liên quan */
.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK {
  display: none !important;
}

/* Ẩn menu chính (MainMenu) */
#MainMenu {
  visibility: hidden !important;
}

/* Ẩn footer */
footer {
  visibility: hidden !important;
}

/* Ẩn header */
header {
  visibility: hidden !important;
}
</style>
"""


# Page title
st.set_page_config(page_title='XG_Boost_Tuned', page_icon='machine-learning-logo.png')
st.markdown(hide_elements_css, unsafe_allow_html=True)
st.title('XGBoost_tuned')

st.sidebar.title("Giới thiệu")
st.sidebar.markdown("""
### Nhóm tác giả:
- **Lý Vĩnh Thuận**
- **Nguyễn Nhựt Trường**
- **Từ Thức**
                    
""")
# Load tuned XGBRegressor model from file
with open('best_gb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)



# Sidebar for accepting input parameters
st.header('1.1. Input data')

st.markdown('**1. Use our clean data**')
df = pd.read_csv("https://raw.githubusercontent.com/ThuanLy-0092/HousePrice_Prediction_Project/main/clean_data.csv", index_col=False)

sleep_time = 1

with st.status("Running ...", expanded=True) as status:

    st.write("Loading data ...")
    time.sleep(sleep_time)

    st.write("Preparing data ...")
    time.sleep(sleep_time)
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    st.write("Splitting data ...")
    time.sleep(sleep_time)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    st.write("Scaling data ...")
    time.sleep(sleep_time)
    # Khởi tạo scaler
    scaler = StandardScaler()

    # Fit scaler trên tập huấn luyện và transform cả tập huấn luyện và tập kiểm tra
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    st.write("Model training ...")
    time.sleep(sleep_time)

    # Use the pre-tuned XGBRegressor model
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)
    
    st.write("Applying model to make predictions ...")
    time.sleep(sleep_time)
        
    st.write("Evaluating performance metrics ...")
    time.sleep(sleep_time)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    st.write("Displaying performance metrics ...")
    time.sleep(sleep_time)
    xgb_results = pd.DataFrame([['Tuned XGBRegressor', train_mse, train_r2, test_mse, test_r2]], columns=['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2'])
    
status.update(label="Status", state="complete", expanded=False)

# Display data info
st.header('Input data', divider='rainbow')
col = st.columns(4)
col[0].metric(label="No. of samples", value=X.shape[0], delta="")
col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")

with st.expander('Initial dataset', expanded=True):
    st.dataframe(df, height=210, use_container_width=True)
with st.expander('Train split', expanded=False):
    train_col = st.columns((3,1))
    with train_col[0]:
        st.markdown('**X**')
        st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
    with train_col[1]:
        st.markdown('**y**')
        st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)
with st.expander('Test split', expanded=False):
    test_col = st.columns((3,1))
    with test_col[0]:
        st.markdown('**X**')
        st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
    with test_col[1]:
        st.markdown('**y**')
        st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)

# Display model parameters
st.header('Model parameters', divider='rainbow')
parameters_col = st.columns(5)
parameters_col[0].metric(label="Test size:", value=0.3, delta="")
parameters_col[1].metric(label="Number of estimators (n_estimators)", value=xgb_model.get_params()['n_estimators'], delta="")
parameters_col[2].metric(label="Max depth (max_depth)", value=xgb_model.get_params()['max_depth'], delta="")
parameters_col[3].metric(label="Regularization alpha (reg_alpha)", value=xgb_model.get_params()['reg_alpha'], delta="")
parameters_col[4].metric(label="Regularization lambda (reg_lambda)", value=xgb_model.get_params()['reg_lambda'], delta="")

# Display feature importance plot
importances = xgb_model.feature_importances_
feature_names = list(X.columns)
forest_importances = pd.Series(importances, index=feature_names)
df_importance = forest_importances.reset_index().rename(columns={'index': 'feature', 0: 'value'})

bars = alt.Chart(df_importance).mark_bar(size=40).encode(
            x='value:Q',
            y=alt.Y('feature:N', sort='-x')
        ).properties(height=250)

# Separate performance and feature importance in different containers
st.header('Model performance', divider='rainbow')
st.dataframe(xgb_results.T.reset_index().rename(columns={'index': 'Parameter', 0: 'Value'}))

st.header('Feature importance', divider='rainbow')
st.altair_chart(bars, theme='streamlit', use_container_width=True)

# Prediction results
st.header('Prediction results', divider='rainbow')
s_y_train = pd.Series(y_train, name='actual').reset_index(drop=True)
s_y_train_pred = pd.Series(y_train_pred, name='predicted').reset_index(drop=True)
df_train = pd.DataFrame(data=[s_y_train, s_y_train_pred], index=None).T
df_train['class'] = 'train'
    
s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
s_y_test_pred = pd.Series(y_test_pred, name='predicted').reset_index(drop=True)
df_test = pd.DataFrame(data=[s_y_test, s_y_test_pred], index=None).T
df_test['class'] = 'test'

df_prediction = pd.concat([df_train, df_test], axis=0)

# Add a multiselect to choose which class to display
class_selection = st.multiselect(
    'Chọn lớp để hiển thị:',
    options=['train', 'test'],
    default=['train', 'test']
)

# Filter the dataframe based on the selected classes
df_filtered = df_prediction[df_prediction['class'].isin(class_selection)]

# Display prediction results and scatter plot in different containers
st.header('Feature importance', divider='rainbow')
st.altair_chart(bars, theme='streamlit', use_container_width=True)

st.header('Prediction results', divider='rainbow')
st.dataframe(df_filtered, height=320, use_container_width=True)

scatter = alt.Chart(df_filtered).mark_circle(size=60).encode(
    x='actual',
    y='predicted',
    color='class'
)
st.altair_chart(scatter, theme='streamlit', use_container_width=True)
