from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from conversation_env.rag import llm, retriever, query, pp
import json


PROMPT_TEMPLATE = """
Human: You are a legal advisor AI system, and provides answers to questions by using fact based and precedent from previous legal cases when possible. 
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use quotations from previous legal cases where possible.

Assistant:"""

claude_prompt = PromptTemplate(template=PROMPT_TEMPLATE, 
                               input_variables=["context","question"])
                               
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": claude_prompt}
)

answer = qa.invoke(query)

filename = "response_2.json"

with open(filename, 'w') as json_file:
    json.dump(answer, json_file, indent=4)
    
with open(filename, 'r') as json_file:
    data = json.load(json_file)
    result = data.get("result", None)

answer_body = answer['result']
pp.pprint(result)