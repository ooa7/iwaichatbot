import streamlit as st
from langchain.chains.qa_with_sources.map_reduce_prompt import question_prompt_template
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorize the sales response csv data
loader = CSVLoader(file_path='iwproject.csv')
documents = loader.load()
#print(documents[0])
#print(len(documents))
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):

    #parameter reduced from 3 (default) to 2 due to db specificity.
    similar_response = db.similarity_search(query, k=2)

    page_contents_array = [doc.page_content for doc in similar_response]
    # print(page_contents_array)
    return page_contents_array
# question = "does abia have an international airport?"
# results = retrieve_info(question)
# print(results)

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
template = """
You are a world class Nigerian historian and economist. 
I will share a question with you and you will give me the best answer in
response to all past questions asked,
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best responses, 
in terms of length, tone of voice, logical arguments and other details, but
always answer in complete sentences

2/ If the best responses are irrelevant, then try to mimic the style of the best responses to the question

Below is a question I received from the customer:
{question}

Here is a list of best responses of how we normally respond to prospect in similar scenarios:
{response}

Please write the best response to the question:
"""
prompt = PromptTemplate(
    input_variables=["question", "response"],
    template=template
)
chain = prompt | llm | StrOutputParser()

# 4. Retrieval augmented generation
def generate_response(question):
    info = retrieve_info(question)
    answer = chain.invoke({"question": question, "response": info})
    return answer
# complex_question = ""
# ans = generate_response(complex_question)
# print(ans)


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="States ChatBot", page_icon="🤖")
    st.header("InfoWARE States of the Nation ChatBot ")
    message = st.text_area("What's your question?")
    if message:
        st.write("Generating response...")
        result = generate_response(message)
        st.info(result)


if __name__ == '__main__':
    main()


