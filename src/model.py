import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

def run_churn_model(retail_silver,rfm):

    # Merge transaction and rfm data
    retail_silver_modeling = retail_silver.merge(rfm,on='CustomerID',how='inner')

    # Features
    feature = retail_silver_modeling[['Recency','Frequency', 'Monetary','is_UK']]

    # Target: 1 means Churned ("At Risk","Churned" Segement), O meas not Churned ("Champions","Promising")
    target = retail_silver_modeling['KMeans_Segment_name'].map(lambda x: 1 if x in ["At Risk","Churned"] else 0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(feature,target,test_size=0.2,random_state=2,stratify=target)

    # Pipeline: Standard Scale + Logistic Regression
    pipe = Pipeline([('scaler',StandardScaler()),('model',LogisticRegression())])

    # Train model
    pipe.fit(X_train,y_train)

    # Predict model
    y_pred = pipe.predict(X_test)
    
    # Evaluation
    print("\n"+"=="*30)
    print("Churn Model Evaluation")
    print("=="*30)
    cm = confusion_matrix(y_test,y_pred)
    tn, fp, fn, tp = cm.ravel().tolist()
    print(f"FPR: {fp/(fp+tn):.0%}")
    print(f"FNR: {fn/(fn+tp):.0%}")
    print(f"TPR: {1-(fn/(fn+tp)):.0%}")
    print("\nClassification Report:")
    print(classification_report(y_test,y_pred))

    return pipe, X_test, y_test, y_pred, cm