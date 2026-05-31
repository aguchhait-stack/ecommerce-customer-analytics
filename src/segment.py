import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans 

def create_segment(rfm):
    rfm_segment = rfm.copy()

    # Log1p transformation for right skwedness
    X = np.log(rfm_segment[["Recency","Frequency","Monetary"]])

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
    rfm_segment.loc[X_train.index,'KMeans_Segment'] = train_labels
    rfm_segment.loc[X_test.index,'KMeans_Segment'] = test_labels
    rfm_segment['KMeans_Segment'] = rfm_segment['KMeans_Segment'].astype(int)

    # Adding Cluster name accoding to business interpretation; Refer notebook for details

    rfm_segment['KMeans_Segment_name'] = rfm_segment['KMeans_Segment']\
                                            .map({2:'Champions',
                                                0:'Promising',
                                                3:"At Risk",
                                                1:"Churned"})

        
    # Summary
    print("=="*25)
    print("KMeans Segment Analysis")
    print("=="*25)
    print(rfm_segment.groupby("KMeans_Segment_name").agg(
       {"Recency"  :  "mean",
        "Frequency":  "mean",
        "Monetary" :  "mean"}
    ).sort_values('Monetary').round(2))

    return rfm_segment


        

