import boto3
import json
from datetime import datetime
from botocore.exceptions import ClientError

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from .base import EngineLM, CachedEngine


def get_completion(prompt, modelId, client, system_prompt=None):
    # Define the inference configuration
    inference_config = {
        "temperature": 0.0,  # Set the temperature for generating diverse responses
        "maxTokens": 500  # Set the maximum number of tokens to generate
    }
    # Define additional model fields
    additional_model_fields = {
        "top_p": 1,  # Set the top_p value for nucleus sampling
    }
    # Create the converse method parameters
    converse_api_params = {
        "modelId": modelId,  # Specify the model ID to use
        "messages": [{"role": "user", "content": [{"text": prompt}]}],  # Provide the user's prompt
        "inferenceConfig": inference_config,  # Pass the inference configuration
        "additionalModelRequestFields": additional_model_fields  # Pass additional model fields
    }
    # Check if system_text is provided
    if system_prompt:
        # If system_text is provided, add the system parameter to the converse_params dictionary
        converse_api_params["system"] = [{"text": system_prompt}]

    # Send a request to the Bedrock client to generate a response
    try:
        response = client.converse(**converse_api_params)

        # Extract the generated text content from the response
        text_content = response['output']['message']['content'][0]['text']

        # Return the generated text content
        return text_content

    except ClientError as err:
        message = err.response['Error']['Message']
        print(f"A client error occured: {message}")

class ChatBedRock(EngineLM, CachedEngine):
    SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string='anthropic.claude-3-sonnet-20240229-v1:0',
        system_prompt=SYSTEM_PROMPT,
    ):
        
        self.session = boto3.Session()
        self.region = "us-west-2"
        print('over here region name')
        print(self.region)
        self.modelId = model_string
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_bedrock_{model_string}.db")
        super().__init__(cache_path=cache_path)
        
        
        self.bedrock_client = boto3.client(service_name = 'bedrock-runtime', region_name = self.region,)
        self.model_string = model_string
        self.system_prompt = system_prompt
        assert isinstance(self.system_prompt, str)
       
       
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))    
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        response = get_completion(prompt, self.modelId, self.bedrock_client, system_prompt=sys_prompt_arg)

        #response = response.content[0].text
        self._save_cache(sys_prompt_arg + prompt, response)
        return response
