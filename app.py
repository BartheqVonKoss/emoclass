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
                     outputs=[gr.outputs.Label(num_top_classes=5), "plot"],
                     examples=[["[NAME] is not fascist. Heck,she's not even alt-right", "LogisticRegression"],
                               ["[NAME] is not fascist. Heck,she's not even alt-right", "RandomForest"],
                               ["Shes awful and its all about her. Try again. And when the show has awful ratings thatll be confirmed.", "LogisticRegression"],
                               ["Shes awful and its all about her. Try again. And when the show has awful ratings thatll be confirmed.", "RandomForest"],
                               ["My favourite food is anything I didn't have to cook myself.", "LogisticRegression"],
                               ["My favourite food is anything I didn't have to cook myself.", "RandomForest"],
                               ["I’m really sorry about your situation :( Although I love the names Sapphira, Cirilla, and Scarlett!", "RandomForest"],
                               ["I’m really sorry about your situation :( Although I love the names Sapphira, Cirilla, and Scarlett!", "LogisticRegression"]])
iface.launch(server_name="0.0.0.0", server_port=1212)
