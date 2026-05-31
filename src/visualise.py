import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-white')
plt.rcParams['figure.dpi'] = 80 

def plot_rfm_distribution(rfm):
    fig,axes = plt.subplots(2,3,figsize=(12,8))
    sns.histplot(x="Recency",data=rfm,color="red",
                kde=True,ax=axes[0,0])
    sns.histplot(x="Frequency",data=rfm,color="green",
                kde=True,ax=axes[0,1])
    sns.histplot(x="Monetary",data=rfm,color="blue",
                kde=True,ax=axes[0,2])
    sns.histplot(x="Recency",data=rfm,color="red",kde=True,
                log_scale=True,ax=axes[1,0])
    sns.histplot(x="Frequency",data=rfm,color="green",kde=True,
                log_scale=True,ax=axes[1,1])
    sns.histplot(x="Monetary",data=rfm,color="blue",
                kde=True,log_scale=True,ax=axes[1,2])
    axes[0,0].set_title('Recency Distribution',fontsize=13)
    axes[0,0].grid(alpha=0.5)
    axes[0,1].set_title('Frequency Distribution',fontsize=13)
    axes[0,1].grid(alpha=0.5)
    axes[0,2].set_title('Monetary Distribution',fontsize=13)
    axes[0,2].grid(alpha=0.5)
    axes[1,0].set_title('Recency Distribution (Log-Scaled)',fontsize=13)
    axes[1,0].grid(alpha=0.5)
    axes[1,1].set_title('Frequency Distribution (Log-Scaled)',fontsize=13)
    axes[1,1].grid(alpha=0.5)
    axes[1,2].set_title('Monetary Distribution (Log-Scaled)',fontsize=13)
    axes[1,2].grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig('outputs/rfm_distributions.png')
    plt.show()



def plot_rfm_segment_heatmap(rfm_segment):

    # Cluster Average
    cluster_avg = rfm_segment.groupby('KMeans_Segment_name').agg({"Recency":"mean","Frequency":"mean","Monetary":"mean"})

    # Population Average
    population_avg = rfm_segment[["Recency", "Frequency", "Monetary"]].mean()

    # Related Importance
    relative_imp = cluster_avg/population_avg - 1

    # Flipped the recency - lower better for meaningful plot
    relative_imp_flipped = relative_imp.copy()
    relative_imp_flipped['Recency'] = -relative_imp['Recency']

    plt.figure(figsize=(10, 6))
    sns.heatmap(relative_imp_flipped,annot=True,cmap='RdYlGn',center=0) # Red<Yellow<Green
    plt.title("Relative RFM Heatmap: Deviation from Dataset Average",fontsize=13)
    plt.xlabel('RFM Attribute')
    plt.ylabel('Segment')
    plt.tight_layout()
    plt.savefig('outputs/rfm_segment_heatmap.png')
    plt.show()

def plot_cohort_heatmap(Retention):

    plt.figure(figsize=(14, 8))
    sns.heatmap(Retention,annot=True,
            fmt='.0%', # annote as % value
            cmap='YlGn',
            vmin=0, # Minimum color floor
            vmax=0.5 # Maximam color ceiling
            ) 
    plt.title('Customer Retention by Cohort (%)', fontsize=13)
    plt.xlabel('Months Since First Purchase')
    plt.ylabel('Cohort Month')  
    plt.tight_layout()
    plt.show()