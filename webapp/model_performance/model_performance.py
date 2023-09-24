from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
import pandas as pd
import  os

PROJECT_ROOT = os.path.dirname(__file__)
def get_performance_as_bokeh() -> object:
    # C:\Users\jpg\Desktop\pythonAItestChrisSakr\webapp\model_performance\unet_data.csv
    # C:\Users\jpg\Desktop\pythonAItestChrisSakr\webapp\model_performance\model_performance\unet_data.csv
    csv_file_path = os.path.join(PROJECT_ROOT, 'unet_data.csv')
    print(f"csv_file_path {csv_file_path}")
    df = pd.read_csv(csv_file_path)


    source = ColumnDataSource(df)


    p = figure(title="Metrics vs. Epoch", x_axis_label="Epoch")


    p.line(x='epoch', y='loss', source=source, line_width=2, legend_label="Loss", color="red")
    p.line(x='epoch', y='lr', source=source, line_width=2, legend_label="Learning Rate", color="blue")
    p.line(x='epoch', y='mean_io_u', source=source, line_width=2, legend_label="Mean IoU", color="green")
    p.line(x='epoch', y='precision', source=source, line_width=2, legend_label="Precision", color="orange")
    p.line(x='epoch', y='recall', source=source, line_width=2, legend_label="Recall", color="purple")

    p.legend.title = "Metrics"
    p.legend.label_text_font_size = "12px"
    return p