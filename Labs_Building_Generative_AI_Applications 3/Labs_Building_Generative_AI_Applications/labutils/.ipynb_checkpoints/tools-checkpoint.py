"""General helper utilities the workshop notebooks"""
# Python Built-Ins:
from io import StringIO
import sys
import textwrap
import os
import ipywidgets as widgets


def print_ww(*args, width: int = 100, **kwargs):
    """Like print(), but wraps output to `width` characters (default 100)"""
    buffer = StringIO()
    try:
        _stdout = sys.stdout
        sys.stdout = buffer
        print(*args, **kwargs)
        output = buffer.getvalue()
    finally:
        sys.stdout = _stdout
    for line in output.splitlines():
        print("\n".join(textwrap.wrap(line, width=width)))

        
def get_models(bedrock_client,
               output_modality = 'TEXT',
               inference_type = 'ON_DEMAND'):

    response = bedrock_client.list_foundation_models(byOutputModality = output_modality, 
                                                     byInferenceType = inference_type)
    
    models = [(item['modelName'], item['modelId']) for item in response['modelSummaries'] if item['modelLifecycle']['status'] == 'ACTIVE']
    
   
    return sorted(models)

def model_selection(bedrock_control_client):
    model_list = get_models(bedrock_control_client)
    model_dict = {name: value for name, value in model_list}
    dropdown = widgets.Dropdown(options=model_dict,
                            description='Select LLM:', 
                            disabled=False,
                            layout=widgets.Layout(width='300px')
                           )
    return dropdown