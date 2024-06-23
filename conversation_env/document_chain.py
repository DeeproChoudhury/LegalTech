from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


from rag import llm, retriever, query, pp
import json



contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)



system_prompt = (
"""
Human: You are a legal advisor AI system, and provides answers to questions by using fact based and precedent from previous legal cases when possible. 
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>
"""
)

initial_context = [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", """<question>
{input}
</question>

The response should be specific and use quotations from previous legal cases where possible.

Assistant:"""),
    ]


claude_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", """<question>
{input}
</question>

The response should be specific and use quotations from previous legal cases where possible.

Assistant:"""),
    ]
)


# PROMPT_TEMPLATE = """
# Human: You are a legal advisor AI system, and provides answers to questions by using fact based and precedent from previous legal cases when possible. 
# Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags. 
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# <context>
# {context}
# </context>

# <question>
# {input}
# </question>

# The response should be specific and use quotations from previous legal cases where possible.

# Assistant:"""

# claude_prompt = PromptTemplate(template=PROMPT_TEMPLATE, 
#                               input_variables=["context","input"])

# history_aware_retriever = create_history_aware_retriever(llm, retriever, claude_prompt)
                               
question_answer_chain = create_stuff_documents_chain(llm, claude_prompt)

chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



conversational_rag_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

answer = conversational_rag_chain.invoke({"input": query}, config={
        "configurable": {"session_id": "abc123"}
    },)["answer"]

print(answer)

answer = conversational_rag_chain.invoke(
    {"input": "Expand on this."},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]
# pp.pprint(answer)

# initial_context.append(("system", answer))

# claude_prompt = ChatPromptTemplate.from_messages(
#     initial_context
# )  

# question_answer_chain = create_stuff_documents_chain(llm, claude_prompt)

# chain = create_retrieval_chain(retriever, question_answer_chain)

# answer = chain.invoke({"input": "Are you sure about this?"})

pp.pprint(answer)

