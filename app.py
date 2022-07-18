import warnings

import gradio as gr
import matplotlib

from infer import run_inference
from src.dataset.utils import get_docs

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

get_docs()
iface = gr.Interface(run_inference,
                     inputs=[gr.inputs.Textbox(lines=4),
                             gr.inputs.Dropdown(choices=["RandomForest", "LogisticRegression"])],
                     outputs=[gr.outputs.Label(num_top_classes=5), "plot"])
                     # examples=[[ "Enjoy the ride!" ]])#, "My favourite food is anything I didn't have to cook myself."])
iface.launch(server_name="0.0.0.0", server_port=1212)
