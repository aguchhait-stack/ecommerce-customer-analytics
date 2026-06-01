# ============================================
# Environment Setup
# ============================================

from src.load_data import load_data_from_zip
from src.preprocess import cleaning 
from src.rfm import create_rfm
from src.segment import create_segment
from src.cohort import create_cohort
from src.model import run_churn_model
from src.visualise import (plot_rfm_distribution, plot_rfm_segment_heatmap, 
                           plot_cohort_heatmap, plot_confusion_matrix, 
                           plot_roc_curve)

if __name__ == "__main__":

    # ============================================
    # 1. DATA WRANGLING
    # ============================================

    retail_bronze = load_data_from_zip()
    retail_silver = cleaning(retail_bronze)

    # ============================================
    # 2. FEATURE ENGINEERING & EDA
    # ============================================

    retention = create_cohort(retail_silver)
    plot_cohort_heatmap(retention)
    rfm = create_rfm(retail_silver)
    plot_rfm_distribution(rfm)


    # ============================================
    # 3. CUSTOMER SEGMENTATION
    # ============================================

    rfm = create_segment(rfm)
    plot_rfm_segment_heatmap(rfm)

    # ============================================
    # 4. CHURN PREDICTION MODEL
    # ============================================

    pipe, X_test,y_test, y_pred, cm = run_churn_model(retail_silver,rfm)
    plot_confusion_matrix(cm)
    plot_roc_curve(pipe,X_test,y_test)
    
    print("🎉 Pipeline executed successfully!")
    print("📊 All outputs and plots saved to the 'outputs/' directory.")
