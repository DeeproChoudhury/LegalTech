
import boto3
import pprint
from botocore.client import Config
import json
from langchain_community.chat_models.bedrock import BedrockChat
from langchain.retrievers.bedrock import AmazonKnowledgeBasesRetriever

##
kb_id = "YL3GYNLP2E"
##
pp = pprint.PrettyPrinter(indent=2)
session = boto3.session.Session()
region = session.region_name
bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
bedrock_client = boto3.client('bedrock-runtime', region_name = region)
bedrock_agent_client = boto3.client("bedrock-agent-runtime",
                              config=bedrock_config, region_name = region)
print(region)
##

def retrieve(query, kbId, numberOfResults=5):
    return bedrock_agent_client.retrieve(
        retrievalQuery= {
            'text': query
        },
        knowledgeBaseId=kbId,
        retrievalConfiguration= {
            'vectorSearchConfiguration': {
                'numberOfResults': numberOfResults,
                'overrideSearchType': "HYBRID", # optional
            }
        }
    )
    
# query = "What is Amazon doing in the field of Generative AI?"
# response = retrieve(query, kb_id, 4)
# retrievalResults = response['retrievalResults']
# pp.pprint(retrievalResults)

# fetch context from the response
def get_contexts(retrievalResults):
    contexts = []
    for retrievedResult in retrievalResults: 
        contexts.append(retrievedResult['content']['text'])
    return contexts

# contexts = get_contexts(retrievalResults)
# pp.pprint(contexts)

modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
accept = 'application/json'
contentType = 'application/json'
llm = BedrockChat(model_id=modelId, 
                  client=bedrock_client)
                  
query = "Provide examples of asylum seekers fearing persecution if deported."
retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        retrieval_config={"vectorSearchConfiguration": 
                          {"numberOfResults": 4,
                           'overrideSearchType': "SEMANTIC",
                           }
                          },
    )
    
 
# docs = retriever.get_relevant_documents(
#         query=query
#     )
# pp.pprint(docs)