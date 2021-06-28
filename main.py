from pandas._config.config import options
from pandas.core.algorithms import value_counts
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
import phik
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from PIL import Image
import re
import itertools
from sklearn.metrics import confusion_matrix
from visualize import *

st.set_page_config(layout="wide")
#import image

intro = Image.open(r'![Loan](https://user-images.githubusercontent.com/81556008/123613501-aea35600-d82d-11eb-9640-79785e7869cb.jpg)')
solution = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\splution process.jpg')
EDA_image = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\exploring.png')
miss_data = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\missdata.jpg')
sol_data = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\solution_data.jpg')
background = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\background.jpg')
edu = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\edu.jpg')
age_ext_1 = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\AGE_EX1.JPG')
ext_2_3 = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\EX2_EX3.JPG')
feature_importance = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\fea_imp.jpg')
machine = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\roboticsjpeg.jpg')
models = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\models.jpg')
smote = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\SMOTE_result.JPG')
undersamp = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\Undersamp_result.JPG')
LGBM_model = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\LGBM.png')
ka_1st = Image.open(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\EDA\Kaggle_1st_place.JPG')

#---------------------------------------------------------------------------------------------------------
#import data
app_data = get_data(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\application_train.csv')
dataset =  pd.read_csv(r"C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\results\train_norm.csv")
X_test = get_data(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\results\X_streamlit.csv')
y_test = get_data(r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\results\y_streamlit.csv')

X_test.columns = list(map(lambda x: str(x).replace(" ", "_").replace("-", "_").replace("_/_", "_").upper(), X_test.columns))
X_test = X_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '',x))

pot_data = dataset[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EXT_SOURCE_2','EXT_SOURCE_3','EXT_SOURCE_1','DAYS_LAST_PHONE_CHANGE','CREDIT_TERM', 'TARGET']]
pot_data.dropna(inplace=True)
#---------------------------------------------------------------------------------------------------------

# create matrix categorical table
categorical_columns = ['TARGET','FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
                                'FLAG_PHONE', 'FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY',
                                'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
                                'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 
                            'LIVE_CITY_NOT_WORK_CITY'] + ['FLAG_DOCUMENT_' + str(i) for i in range(2,22)] + app_data.dtypes[
                                app_data.dtypes == 'object'].index.tolist()

table_cat = plot_phik_matrix(app_data, categorical_columns, figsize = (15,15), fontsize = 8)

#---------------------------------------------------------------------------------------------------------

#create numerical table
columns_to_drop = ['SK_ID_CURR'] + list(set(categorical_columns) - set(['TARGET']))
table_con = target_top_corr(app_data,columns_to_drop)


#000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 

def home(mode):
    
    header = st.beta_container()


    # Web App Title
    with header:
        st.title("CLIENT CAPABLE OF REPAYMENT PREDICTION")
        st.text('Prepared by: Vuong Vu Thu Ha _ A person is really good at creating bugs >_<')
        st.image(intro)
        st.header('Why this project?')
        st.markdown('* According to first-quarter financial statements released by 26 banks, bad debts of these banks increased by 5.3 per cent to more than $4 billion.')
        
        st.header('The goal of project')
        st.markdown('* Predict who is **Payer** (who has difficulties to repay the loans) versus **Non_Payer** (who has full potentail to repay loans on time)')
        st.markdown('* Extract reasons for the prediction result of each client')
        
        st.header('Solution Process')  
        st.image(solution) 


#000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000      

def dashboard (mode):

    st.title('Data exploration')
    st.image(EDA_image)
    c1,c2,c3 = st.beta_columns((1,1,1))
    c1.header('Problems in data!')
    c1,c2 = st.beta_columns((1,1))
    c1.write('**Missing data reduces statistical power**')
    c1.image(miss_data)
    target_distribution = app_data.TARGET.value_counts()
    labels = ['Payer', 'Non_Payer']
    fig = go.Figure(data = [
        go.Pie(values = target_distribution, labels = labels, textinfo = 'label+percent' , pull = [0,0.04])])
    c2.write('**Imbalanced dataset makes model be biased towards the majority class.**')
    c2.plotly_chart(fig)

    c1,c2,c3 = st.beta_columns((1,2,1))
    c2.header('Data solution process')
    c2.image(sol_data)

    c2.write('')
    c2.write('')
    c2.subheader('Client_Background and Credit_history have effect to label')
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(pot_data.values)
    pot_data['others'] = pca_result[:,0]
    pot_data['credit_history'] = pca_result[:,1] 
    pot_data['background'] = pca_result[:,2]

    import plotly.express as px
    fig = px.scatter_3d(pot_data, x='others', y='credit_history', z='background',
              color='TARGET')
    c2.write(fig)
    

    #-----------------------------------------------------------------------------------------------
    
    st.subheader("Background affects mainly the ability of repayment ") 
    c1,c2 = st.beta_columns((2,1))
    #plotting overall distribution of category  
    c1.image(background) 
    c2.write('')
    c2.write('')
    c2.image(edu)

    st.subheader("Age and Credit_Score impact to label!") 
    st.image(age_ext_1)
    st.image(ext_2_3)

    st.subheader('Solve imbalanced classes issue')
    sel_col, disp_col = st.beta_columns(2)
    sel_col.subheader('Oversampling')
    sel_col.image(smote)

    disp_col.subheader('Undersampling')
    disp_col.image(undersamp)
    


#000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000   

def model(mode):
    
    st.title('Analyze model')
    st.text('What behind the scence?')
    st.image(machine)
    
    c1,c2,c3 = st.beta_columns((2,1,1))
    c1.header('Model comparision')
    c1.image(models)

    gbm = lgb.Booster(model_file=r'C:\Users\Duong Nguyen\Desktop\final_project\home-credit-default-risk\results\my_model.txt')
    probs = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    c2.header('The model accuracy')
    # calculate AUC
    auc = roc_auc_score(y_test, probs)
    c2.write('LGBM AUC: {:.4f}'.format(auc))

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)

    # plot the roc curve for the model
    ROC_fig = plt.figure()
    plt.plot(fpr, tpr, marker='.',label='AUC_score')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate') 

    # show the plot
    plt.legend()
    c2.write(ROC_fig)
    c3.write('')
    c3.write('')
    c3.write('')
    c3.write('')
    c3.write('')
    c3.write('')
    c3.write('')
    c3.image(ka_1st)
    
    #*****************************************************************
    
    st.subheader('Feature importance')
    st.image(feature_importance)
    
    import xgboost
    import shap
    import streamlit.components.v1 as components


    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    # train XGBoost model
    bst = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X_test, label=y_test), 100)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X_test)
    st.subheader('What make wrong decision?')
    prediction = bst.predict(xgboost.DMatrix(X_test))
    predict = []
    for i in prediction:
        if i > 0.5:
            i = 1
            predict.append(i)
        else:
            i = 0
            predict.append(i)


    # visualize the first prediction's explaination with default colors
    c1,c2,c3,c4 = st.beta_columns((1,1,1,1))
    number_client = c1.text_input('The number of loan','2')
    number_client = int(number_client)
    c2.write('The predicted value')
    if predict[number_client] == 0:
        c2.text('Payer')
    else:
        c2.text('Non_Payer')
    c3.write('The true value')
    if y_test.values[number_client][0] == 0:
        c3.text('Payer')
    else:
        c3.text('Non_Payer')

    visual = st_shap(shap.force_plot(explainer.expected_value, shap_values[number_client,:], X_test.iloc[number_client,:]))
    st.write(visual)

   
#---------------------------------------------------------------------------------------------------------
def main():
    main_mode = st.sidebar.selectbox('Select Menu',
    ['Home','Data Exploration','Model'])

    if main_mode == 'Home':
        home(main_mode)
    elif main_mode == 'Data Exploration':
        dashboard(main_mode)
    elif main_mode == 'Model':
        model(main_mode)

if __name__ == '__main__':
    main()

