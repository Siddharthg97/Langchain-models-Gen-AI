import pprint
import streamlit as st
import pandas as pd
import numpy as np
from langchain_core.tools import tool
from langchain.tools import Tool
import os

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from google.cloud import bigquery
from langchain_openai import AzureChatOpenAI
from langchain_google_vertexai import VertexAI
import warnings
warnings.filterwarnings(action='ignore')

from google.cloud import bigquery


# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'training-422706-5d43eac15616.json'

st.set_page_config(
    page_title="SQL Query Generator",
    page_icon='tredence-squareLogo-1650044669923.webp',
)

@st.cache_resource
def load_bigquery(uri):
    db = SQLDatabase.from_uri(uri)
    return db

db = load_bigquery("bigquery://training-422706")

client = bigquery.Client()

def convert_bq_iterator_to_dataframe(iterator):
    rows = []
    for row in iterator:
        rows.append(list(row))  # Convert row to list
    if not rows:
        return None

    column_names = [field.name for field in iterator.schema]
    df = pd.DataFrame(rows, columns=column_names)
    return df

@tool
def get_data_dictionary(table_name: str) -> str:
    """
    Returns the data dictionary for the provided table name
    input: the table name about which you want to get info
    """
    name = table_name.split('_Data_Dict')[0] + '_Data_Dict'
    query_job = client.query(f'SELECT * FROM {name}')  # API request
    rows = query_job.result()  # Waits for query to finish
    df = convert_bq_iterator_to_dataframe(rows)
    return df.to_string(index = False) 
    
tools = [
    Tool(
        name="get_data_dictionary",
        description="Returns the data dictionary for the provided table name.",
        func=get_data_dictionary
    )
]


st.title('Walmart Table Query Tool')
question=st.text_input("Input: ",key="input")


st.write('\n')
st.markdown('##### Records to Return (default: 100)')
no_of_records = st.slider('hour', 10, 10000, 100)

submit=st.button("Ask the question")
if submit:
    import base64
    import regex as re


    def generate():
        llm = VertexAI(model_name = "gemini-1.5-flash-preview-0514")
            # If the question does not seem related to the database, just return "I don't know" as the answer."""

        agent_executor = create_sql_agent(llm, db=db, verbose=True, agent_type="openai-tools",
        extra_tools = tools)
        
        response = agent_executor.invoke(
            {
                "input": f"""You are an agent designed to interact with bigquery database.
                Given an input question, create a syntactically correct bigquery query to run.
                
                give me the bigquery sql query - {question}. Return only the final bigquery query and not the results.
                You can use get_data_dictionary tool that is provided to you to know about the column meanings in a table before starting the solution
                
                Only use the columns present in the tables and don't assume any information.

                Only the SQL query is needed. Dont try to find the result also.
                
                Tables must be qualified with a dataset"""
            }
                        # If you need clarification on meaning of columns in a table, you can access the 
                # data dictionary(if available) from database tables named "<table_name>_data_dictionary".
        )
        return response['output']

    text = generate()
    
    st.subheader('Query')
    st.write(text)

    def extract_sql_query(document):
        # Regular expression to match text within triple backticks containing 'sql'
        pattern = r"```sql\s+(.*?)\s+```"
        
        # Find all matches in the document
        matches = re.findall(pattern, document, re.DOTALL)
        
        return matches

    print(text)
    

    # Perform a query.
    test_query = extract_sql_query(text)
    if not test_query:
        test_query = [text]
    if 'limit' not in test_query[0].lower():
        test_query = [test_query[0].replace(";", "") + " LIMIT " + str(no_of_records)+ ';']
    QUERY = (test_query[0])

    try:
        # st.write(QUERY)
        query_job = client.query(QUERY)  # API request
        rows = query_job.result()  # Waits for query to finish

        df = convert_bq_iterator_to_dataframe(rows)

        # st.write(QUERY)

        ## Streamlit

        #set page header

        st.subheader('Result')
        st.write(df)
    except:
        st.write('A valid bigquery could not be obtained. Please check the question.')