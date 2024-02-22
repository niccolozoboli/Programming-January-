import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import clean_file as cl


df=pd.read_csv('Salaries.csv', low_memory=False)
#set the cmap for the plots
pastel1=plt.get_cmap('Pastel1')
st.title("San Francisco's Government public jobs salaries")


#create sidebar to access various part of the project
st.sidebar.write('What do you want to see?')


#create a button in the sidebar for the EDA part 
if st.sidebar.checkbox('Exploratory Data Analysis'):
    st.header('Exploratory Data Analysis')
    
    #create a button to display the dataset informations before data cleaning
    if st.checkbox('Before Dataset Cleaning'):
        st.write(df)
        '''Columns' names:'''
        #with a list comprehension transform the index output of df.columns in a list of strings to be displayed
        st.write(str([str(x) for x in df.columns]).replace(',',', ').replace("'",'').strip('[').strip(']')+'.')
        st.write('How many Rows/Columns:', df.shape[0],',', df.shape[1], '.')
        st.write('Dataset head:', df.head())
        st.write('Dataset tail:', df.tail())
        st.write('Some statistics about the wages')
        st.write('Mean TotalPayBenefits: ', round(df['TotalPayBenefits'].mean()), '$')
        st.write('Median TotalPayBenefits: ', round(df['TotalPayBenefits'].median()), '$')
        st.write('Mode TotalPayBenefits: ', round(df['TotalPayBenefits'].mode()[0]), '$')
        st.write('Minimum TotalPayBenefits: ', round(df['TotalPayBenefits'].min()), '$')
        st.write('Maximum TotalPayBenefits: ', round(df['TotalPayBenefits'].max()), '$')
        st.write('Standard deviation of TotalPayBenefits: ', round(df['TotalPayBenefits'].std()), '$')
        st.write('How many null values per column?', pd.DataFrame(df.isnull().sum(), columns=['Null values']).T)
        pastel1=plt.get_cmap('Pastel1')
        graph=plt.figure(figsize = (7, 5))
        plt.title('Null Values per Column', fontsize=30, pad=15)
        plt.barh(y=df.isnull().sum().index, width=df.isnull().sum(), color=pastel1.colors, edgecolor='black')
        plt.ylabel('Columns') 
        plt.xlabel('Count')
        st.write(graph)
        
    #create a button to display the dataset informations after data cleaning
    if st.checkbox('After Dataset Cleaning'):
        #use the clean_file package to clean the dataframe with cl.clean_df
        cl_df=cl.clean_df(df)
        st.write(cl_df)
        '''Columns' names:'''
        #with a list comprehension transform the index output of df.columns in a list of strings to be displayed
        st.write(str([str(x) for x in cl_df.columns]).replace(',',', ').replace("'",'').strip('[').strip(']')+'.')
        st.write('How many Rows/Columns:', cl_df.shape[0],',', cl_df.shape[1], '.')
        st.write('Dataset head:', cl_df.head())
        st.write('Dataset tail:', cl_df.tail())
        st.write('Some statistics about the wages')
        st.write('Mean TotalPayBenefits: ', round(cl_df['TotalPayBenefits'].mean()), '$')
        st.write('Median TotalPayBenefits: ', round(cl_df['TotalPayBenefits'].median()), '$')
        st.write('Mode TotalPayBenefits: ', round(cl_df['TotalPayBenefits'].mode()[0]), '$')
        st.write('Minimum TotalPayBenefits: ', round(cl_df['TotalPayBenefits'].min()), '$')
        st.write('Maximum TotalPayBenefits: ', round(cl_df['TotalPayBenefits'].max()), '$')
        st.write('Standard deviation of TotalPayBenefits: ', round(cl_df['TotalPayBenefits'].std()), '$')
        #create a dataframe of the null values per column, transposed so it is horizontal
        st.write('How many null values per column?', pd.DataFrame(cl_df.isnull().sum(), columns=['Null values']).T)


#use the clean_file package to clean the dataframe with cl.clean_df, since for the rest of the code the df needs to be cleaned
df=cl.clean_df(df)


#create a button in the sidebar for the plots part 
if st.sidebar.checkbox('Data Visualization'):
    st.header('Data Visualization')
    #create a drop down menu to select the graph to display
    plot_sbox = st.selectbox('Which graph would you like to be see?', ['How many samples per year', 'Average TotalPayBenefits per year', 'Standard deviation for TotalPayBenefits per year', 'How many employees in Fire/Police departments', 'Distributions of numeric columns', 'Correlation between variables', 'Correlation between Status and other numeric variables', 'Most frequent jobs', 'Top 10 jobs by average BasePay/TotalPayBenefits', 'Bottom 10 jobs by average TotalPayBenefits'])
     
    if plot_sbox=='How many samples per year':
        graph=plt.figure(figsize=(5,5))
        plt.title('How many samples per year', fontsize=25, pad=20)
        plt.pie(df['Year'].value_counts(), autopct='%.2f%%', colors=pastel1.colors, wedgeprops={'edgecolor':'black'}, shadow=True)
        plt.legend(labels=df['Year'].value_counts().index)
        st.pyplot(graph)
        
    if plot_sbox=='Average TotalPayBenefits per year':
        mean_pay_per_year=df.groupby('Year')['TotalPayBenefits'].mean()
        graph=plt.figure(figsize=(7, 4))
        plt.title('Average TotalPayBenefits per year', fontsize=28, pad=20)
        plt.bar(x=mean_pay_per_year.index, height=mean_pay_per_year, color=pastel1.colors, edgecolor='black')
        plt.xticks(range(2011, 2015))
        plt.xlabel('Year')
        plt.ylabel('Average Total Pay+Benefits')
        st.pyplot(graph)
        
    if plot_sbox=='Standard deviation for TotalPayBenefits per year':
        std_pay_per_year=df.groupby('Year')['TotalPayBenefits'].std()
        graph=plt.figure(figsize=(7, 4))
        plt.title('Standard deviation for TotalPayBenefits per year', fontsize=20, pad=20)
        plt.bar(x=std_pay_per_year.index, height=std_pay_per_year, color=pastel1.colors, edgecolor='black')
        plt.xticks(range(2011, 2015))
        plt.xlabel('Year')
        plt.ylabel('Std for Total Pay+Benefits')
        st.pyplot(graph)
        
    if plot_sbox=='How many employees in Fire/Police departments':
        #create a mask for the fire department and the police department
        fire_mask=df['JobTitle'].str.contains('Fire', case=False)
        police_mask=df['JobTitle'].str.contains('Police', case=False)
        #with the masks, create two separate dfs with only fire department employees and police department employees, and then concatenate them
        fire_df=pd.DataFrame(df[fire_mask], columns=df.columns)
        police_df=pd.DataFrame(df[police_mask], columns=df.columns)
        fire_police_df=pd.concat([fire_df, police_df], axis=0, ignore_index=False, join='outer')
        fire_police_df.reset_index(drop=True, inplace=True)
        fire_police_df['Fire_Dep']=fire_police_df['JobTitle'].str.contains('Fire', case=False)
        graph=plt.figure(figsize=(7,7))
        plt.title('How many employees in Fire/Police departments', fontsize=20, pad=20)
        plt.pie(fire_police_df['Fire_Dep'].value_counts(), autopct='%.2f%%', colors=pastel1.colors, wedgeprops={'edgecolor':'black'}, shadow=True, startangle=90)
        plt.legend(labels=['Police Department', 'Fire Department'], loc='upper left')
        st.pyplot(graph)
        
    if plot_sbox=='Distributions of numeric columns':
        #create a drop down menu to select the column to use for the distribution graph
        distr_sbox=st.selectbox('What column?', ['Empty', 'BasePay', 'OvertimePay', 'OtherPay', 'Benefits', 'TotalPay', 'TotalPayBenefits'])
        if distr_sbox=='Empty':
            pass        
        if distr_sbox=='BasePay':
            x_ticks=[str(x)+'$' for x in range(0,300001,50000)]
            graph=plt.figure(figsize=(10, 4))
            plt.title('Distribution of BasePay', fontsize=30, pad=20)
            plt.hist(df['BasePay'], bins=50, color=pastel1.colors[0], edgecolor='Black')
            plt.xlabel('Dollars')
            plt.ylabel('Count')
            plt.xticks(range(0,300001,50000), labels=x_ticks)
            st.pyplot(graph)            
        if distr_sbox=='OvertimePay':
            x_ticks=[str(x)+'$' for x in range(0,250001,25000)]
            graph=plt.figure(figsize=(10, 4))
            plt.title('Distribution of OverTimePay', fontsize=30, pad=20)
            plt.hist(df['OvertimePay'], bins=30, color=pastel1.colors[0], edgecolor='Black')
            plt.xlabel('Dollars')
            plt.ylabel('Count')
            plt.xticks(range(0,250001,25000), labels=x_ticks)
            st.pyplot(graph)            
        if distr_sbox=='OtherPay':
            x_ticks=[str(x)+'$' for x in range(0,400001,50000)]
            graph=plt.figure(figsize=(10, 4))
            plt.title('Distribution of OtherPay', fontsize=30, pad=20)
            plt.hist(df['OtherPay'], bins=40, color=pastel1.colors[0], edgecolor='Black')
            plt.xlabel('Dollars')
            plt.ylabel('Count')
            plt.xticks(range(0,400001,50000), labels=x_ticks)
            st.pyplot(graph)           
        if distr_sbox=='Benefits':
            x_ticks=[str(x)+'$' for x in range(0,100001,20000)]
            graph=plt.figure(figsize=(10, 4))
            plt.title('Distribution of the Benefits', fontsize=30, pad=20)
            plt.hist(df['Benefits'], bins=30, color=pastel1.colors[0], edgecolor='Black')
            plt.xlabel('Dollars')
            plt.ylabel('Count')
            plt.xticks(range(0,100001,20000), labels=x_ticks)
            st.pyplot(graph)           
        if distr_sbox=='TotalPay':
            x_ticks=[str(x)+'$' for x in range(0,500001,100000)]
            graph=plt.figure(figsize=(10, 4))
            plt.title('Distribution of the TotalPay', fontsize=30, pad=20)
            plt.hist(df['TotalPay'], bins=50, color=pastel1.colors[0], edgecolor='Black')
            plt.xlabel('Dollars')
            plt.ylabel('Count')
            plt.xticks(range(0,500001,100000), labels=x_ticks)
            st.pyplot(graph)            
        if distr_sbox=='TotalPayBenefits':
            x_ticks=[str(x)+'$' for x in range(0,500001,100000)]
            graph=plt.figure(figsize=(10, 4))
            plt.title('Distribution of the TotalPayBenefits', fontsize=30, pad=20)
            plt.hist(df['TotalPayBenefits'], bins=50, color=pastel1.colors[0], edgecolor='Black')
            plt.xlabel('Dollars')
            plt.ylabel('Count')
            plt.xticks(range(0,500001,100000), labels=x_ticks)
            st.pyplot(graph)
            
    if plot_sbox=='Correlation between variables':
        df_corr=df[['BasePay', 'OvertimePay', 'OtherPay', 'Benefits', 'TotalPay', 'TotalPayBenefits']].corr()
        graph=plt.figure()
        plt.title('Correlation between variables', fontsize=25, pad=20)
        sns.heatmap(df_corr, cmap='GnBu', linecolor='black', linewidths=0.5)
        plt.xticks(rotation=45)
        st.pyplot(graph)
        
    if plot_sbox=='Correlation between Status and other numeric variables':
        df['Status_Enc']=df['Status']
        df['Status_Enc']=df['Status_Enc'].replace({'PT':0, 'FT':1})
        df_corr=df[['BasePay', 'OvertimePay', 'OtherPay', 'Benefits', 'TotalPay', 'TotalPayBenefits', 'Status_Enc']].corr()
        graph=plt.figure(figsize=(10,3))
        plt.title('Correlation between Status and other numeric variables', fontsize=18, pad=15)
        sns.heatmap(df_corr[['Status_Enc']][:-1].T, cmap='GnBu', linecolor='black', linewidths=0.5, annot=True)
        plt.yticks([0.5], labels=['Status'], rotation=0)
        st.pyplot(graph)
        
    if plot_sbox=='Most frequent jobs':
        frequent_10_jobs = df['JobTitle'].value_counts()[:9].sort_values()
        graph=plt.figure(figsize=(7, 4))
        plt.title('Most frequent jobs', fontsize=30, pad=20, x=0.26)
        plt.barh(y=frequent_10_jobs.index, width=frequent_10_jobs, color=pastel1.colors, edgecolor='black')
        plt.xlabel('Count', fontsize=20)
        plt.ylabel('Jobs', fontsize=20, labelpad=20)
        st.pyplot(graph)
        
    if plot_sbox=='Top 10 jobs by average BasePay/TotalPayBenefits':
        top_10_jobs_basepay = df.groupby('JobTitle')['BasePay'].mean().sort_values()[-10:]
        top_10_jobs_tpbenefits = df.groupby('JobTitle')['TotalPayBenefits'].mean().sort_values()[-10:]
        x_ticks=[str(x)+'$' for x in range(0, 500001, 50000)]
        graph,axs=plt.subplots(2,1,figsize=(10,14))
        plt.subplots_adjust(hspace=0.4)
        axs[0].set_title('Top 10 jobs by average BasePay', fontsize=30, pad=20, x=0.25)
        axs[0].barh(y=top_10_jobs_basepay.index, width=top_10_jobs_basepay, color=pastel1.colors, edgecolor='black')
        axs[0].set_xlabel('Average BasePay', fontsize=15)
        axs[0].set_ylabel('Jobs', fontsize=15, labelpad=85)
        axs[0].set_xticks(ticks=range(0, 500001, 50000), labels=x_ticks)
        axs[1].set_title('Top 10 jobs by average TotalPayBenefits', fontsize=30, pad=20, x=0.25)
        axs[1].barh(y=top_10_jobs_tpbenefits.index, width=top_10_jobs_tpbenefits, color=pastel1.colors, edgecolor='black')
        axs[1].set_xlabel('Average TotalPayBenefits', fontsize=15)
        axs[1].set_ylabel('Jobs', fontsize=15, labelpad=20)
        axs[1].set_xticks(ticks=range(0, 500001, 50000), labels=x_ticks)
        st.pyplot(graph)
        
    if plot_sbox=='Bottom 10 jobs by average TotalPayBenefits':
        bot_10_jobs_tpbenefits = df.groupby('JobTitle')['TotalPayBenefits'].mean().sort_values()[:10]
        x_ticks=[str(x)+'$' for x in range(0, 500001, 50000)]
        graph=plt.figure(figsize=(7, 4))
        plt.title('Bottom 10 jobs by average TotalPayBenefits', fontsize=30, pad=20, x=0.16)
        plt.barh(y=bot_10_jobs_tpbenefits.index, width=bot_10_jobs_tpbenefits, color=pastel1.colors, edgecolor='black')
        plt.xlabel('Average TotalPayBenefits', fontsize=15)
        plt.ylabel('Jobs', fontsize=15, labelpad=20)
        st.pyplot(graph)


#create a button in the sidebar for the prediction part 
if st.sidebar.checkbox('Prediction on Status'):
    #create the Status_Enc column
    df['Status_Enc']=df['Status']
    df['Status_Enc']=df['Status_Enc'].replace({'PT':0, 'FT':1})
    st.write('In order to use a machine learning technique to predict the status of the single employee, we need to encode the "Status" column.')
    st.write('Full-Time will be represented with a 1 and Part-Time with a 0.')
    
    #create a subheader for the model training part
    st.subheader('Model training')
    st.write('The classification model used is Logistic Regression.')
    st.write('Since we only have', str(df['Status'].value_counts().sum()), 'values in the Status column out of', str(len(df['Status'])), 'rows, this means we have', str(df['Status'].isnull().sum()), 'values to be filled with the prediction model.')
    st.write('This also means that we have', str(df['Status'].value_counts().sum()), 'values to train our Logistic Regression.')
    st.write('With the following slider we can manually select the size of the training data and the test data.')
    #create a slider to manually select the test size to use
    tsize=st.slider('Choose test size', 10, 90, step=10)
    #use the clean_file package to undersample the dataframe with cl.rus_df and to drop the non-numeric columns
    df_num=cl.rus_df(df)
    X=df_num.drop('Status_Enc', axis=1)
    y=df_num['Status_Enc']
    X_tr, X_te, y_tr, y_te=train_test_split(X, y, test_size=tsize/100)
    st.write('Size of the training data:', str(len(X_tr)))
    st.write('Size of the test data:', str(len(X_te)))
    clf=LogisticRegression()
    clf.fit(X_tr, y_tr)
    y_pred=clf.predict(X_te)
    
    #create a subheader for the model evaluation part
    st.subheader('Model evaluation')
    st.write(f'Classification report for the Logistic Regression trained on {100-tsize}% of the available data.')
    #create 3 columns to center the classification report
    col_one, col_two, col_three=st.columns([2.3, 6, 1])
    #save the classification report as a dictionary in the cl_rep variable
    cl_rep=classification_report(y_te, y_pred, output_dict=True)
    #delete the accuracy key in the cl_rep dictionary
    del cl_rep['accuracy']
    #create a dataframe to display the cl_rep dictionary
    col_two.write(pd.DataFrame(cl_rep).round(2).T)
    st.write(f'Accuracy is around {round(accuracy_score(y_te, y_pred)*100, 2)}%.')
    
    #create a subheader for the model training part
    st.subheader('Filling the missing data')
    st.write("Using our now trained model, we can fill the missing values of the Status column for those employees that aren't registered as Full-Time or Part-Time.")
    X=df.drop(['Status', 'Status_Enc', 'EmployeeName', 'JobTitle', 'Year'], axis=1)
    df['Status_Enc']=clf.predict(X)
    df['Status']=df['Status_Enc'].replace({0:'PT', 1:'FT'})
    df=df.drop('Status_Enc', axis=1)
    graph=plt.figure(figsize=(7,5))
    plt.title('Number of Full-Time and Part-Time workers', fontsize=20, pad=20)
    plt.bar(x=df['Status'].value_counts().index, height=df['Status'].value_counts(), color=pastel1.colors, edgecolor='black')
    plt.xticks([0,1], labels=['Part-Time', 'Full-Time'])
    st.pyplot(graph)