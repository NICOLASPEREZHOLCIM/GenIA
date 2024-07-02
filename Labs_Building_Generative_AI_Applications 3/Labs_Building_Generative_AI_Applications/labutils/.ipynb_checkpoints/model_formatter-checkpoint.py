import json
# Model IDs Reference:  https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html

# Amazon titan text reference https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
def titan_text_body_builder(user_input,
                            temperature,
                            topP,
                            maxTokenCount,
                            stopSequences):
    
    body = json.dumps({
    "inputText": "Command: " + user_input,
    "textGenerationConfig":{
        "maxTokenCount":maxTokenCount,
        "stopSequences":stopSequences,
        "temperature":temperature,
        "topP":topP
        }
    })
    
    return body

def claude_body_builder(user_input,
                        role,
                        system,
                        max_tokens,
                        temperature,
                        topP,
                        topK,
                        stopSequences,
                        image_data):
    
    body = {
            "anthropic_version": "bedrock-2023-05-31",    
            "max_tokens": max_tokens,
            "system": system,    
            "messages": [
                {
                    "role": role,
                    "content": [
                            {"type": "text", "text": user_input}
                    ]
                }
            ],
            "temperature": temperature,
            "top_p": topP,
            "top_k": topK,
#             "tools": [
#                 {
#                         "name": string,
#                         "description": string,
#                         "input_schema": json

#                 }
#             ],
#             "tool_choice": {
#                 "type" :  string,
#                 "name" : string,
#             },
            "stop_sequences": stopSequences
        }
    
    if image_data:
        body["messages"][0]["content"].append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}})
    
    return json.dumps(body)

def ai21_body_builder(user_input,
                      temperature,
                      topP,
                      maxTokenCount,
                      stopSequences,
                      penalties):
    body = {
            "prompt": user_input,
            "temperature": temperature,
            "topP": topP,
            "maxTokens": maxTokenCount,
            "stopSequences": stopSequences
            # "countPenalty": {
            #     "scale": float
            # },
            # "presencePenalty": {
            #     "scale": float
            # },
            # "frequencyPenalty": {
            #     "scale": float
            # }
        }
    if penalties:
        body.append(penalties)
    
    return json.dumps(body)

def llama_body_builder(user_input, system, temperature, topP, maxTokenCount):
    
    prompt = "<s>[INST] <<SYS>> {{ " + system + " }} <</SYS>> {{ " + user_input + "  }} [/INST]</s>"
    
    body = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": topP,
        "max_gen_len": maxTokenCount
    }
    
    return json.dumps(body)
    
def model_body_builder(modelid  = 'amazon.titan-text-express-v1',
                               role = 'user',
                               system = 'You are an AI assistant designed to be helpful, harmless, and honest. Your goal is to  provide informative and substantive responses to queries while avoiding potential harms. Skip the preamble and provide only the results. Do not assume anything',
                               user_input  = 'Give me instructions on how to use Amazon Bedrock runtime API',                                             
                               temperature = 0.7,
                               topP = 0.9,
                               topK = 50,
                               maxTokenCount = 512,
                               stopSequences = [],
                               image_data = None,
                               penalties = None):
    
    modelid = modelid.lower()
    
    if 'amazon' in modelid:
        return titan_text_body_builder(user_input, temperature, topP, maxTokenCount, stopSequences)
    elif 'anthropic' in modelid:
        return claude_body_builder(user_input, role, system, maxTokenCount, temperature, topP, topK, stopSequences, image_data)
    elif 'ai21' in modelid:
        return ai21_body_builder(user_input, temperature, topP, maxTokenCount, stopSequences, penalties)
    elif 'meta' in modelid:
        return llama_body_builder(user_input, system, temperature, topP, maxTokenCount)
        
    else:
        return "modelid " + modelid + " is currently not supported"
        
    
## #################    
## Models response parsers

def titan_text_response_parser(response_body):
    return response_body.get('results')[0].get('outputText')

def claude_response_parser(response_body):
    return response_body.get('content')[0].get('text')

def ai21_response_parser(response_body):
    return response_body.get('completions')[0].get('data').get('text')

def meta_response_parser(response_body):
    return response_body['generation']

       
def model_response_parser(modelid, response):
    response_body = json.loads(response.get('body').read())
    
    modelid = modelid.lower()
    
    if 'amazon' in modelid:
        return titan_text_response_parser(response_body)
    elif 'anthropic' in modelid:
        return claude_response_parser(response_body)
    elif 'ai21' in modelid:
        return ai21_response_parser(response_body)
    elif 'meta' in modelid:
        return meta_response_parser(response_body)
    else:
        return "modelid " + modelid + " is currently not supported"