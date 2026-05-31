from src.load_data import load_data_from_zip
from src.preprocess import cleaning 
from src.rfm import create_rfm
from src.segment import create_segment
from src.cohort import create_cohort
from src.visualise import plot_rfm_distribution
from src.visualise import plot_rfm_segment_heatmap
from src.visualise import plot_cohort_heatmap


retail_bronze = load_data_from_zip()

retail_silver = cleaning(retail_bronze)

Retention = create_cohort(retail_silver)

rfm = create_rfm(retail_silver)

rfm_segment = create_segment(rfm)


plot_rfm_distribution(rfm)

plot_rfm_segment_heatmap(rfm_segment)

plot_cohort_heatmap(Retention)