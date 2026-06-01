import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans 

def create_segment(rfm):
    rfm = rfm.copy()

    # Log1p transformation for right skwedness
    X = np.log(rfm[["Recency","Frequency","Monetary"]])

    # train-test split with 80% train set, random state for reproducibility
    X_train,X_test = train_test_split(X,test_size=0.2,random_state = 42)

    # Initiating pipeline object
    pipe = Pipeline([('scaler',StandardScaler()),('kmeans',KMeans(n_clusters = 4,random_state = 42))])
    # fit the pipe with training set
    pipe.fit(X_train)

    # Predicted labels on train dataset
    train_labels = pipe.named_steps['kmeans'].labels_

    # Predicted labels on train dataset
    test_labels = pipe.predict(X_test)

    # assign labels back to rfm tables with right index
    rfm.loc[X_train.index,'KMeans_Segment'] = train_labels
    rfm.loc[X_test.index,'KMeans_Segment'] = test_labels
    rfm['KMeans_Segment'] = rfm['KMeans_Segment'].astype(int)

    # Adding Cluster name accoding to business interpretation
    # Refer the noetbook.ipynb for rationale

    rfm['KMeans_Segment_name'] = rfm['KMeans_Segment']\
                                            .map({2:'Champions',
                                                0:'Promising',
                                                3:"At Risk",
                                                1:"Churned"})

        
    # Summary
    print("=="*25)
    print("KMeans Segment Analysis")
    print("=="*25)
    print(rfm.groupby("KMeans_Segment_name").agg(
       {"Recency"  :  "mean",
        "Frequency":  "mean",
        "Monetary" :  "mean"}
    ).sort_values('Monetary').round(2))

    return rfm


        

