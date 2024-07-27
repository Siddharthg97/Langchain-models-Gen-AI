#!/usr/bin/env python
# coding: utf-8

# ## Install and import necessary libraries

# In[ ]:


# pip install -q langchain


# In[ ]:


import os
import pandas as pd
import re


# ## Function definitions

# In[ ]:


# loading CSV file as LangChain Document
def load_document(file):
    import os
    from langchain.document_loaders import CSVLoader
    loader = CSVLoader(file)#, encoding="utf-8")
    data = loader.load()
    return data


# In[ ]:


# Chunking documents for faster processing
def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=3)
    chunks = text_splitter.split_documents(data)
    return chunks


# ## Reading input data

# In[ ]:


df = pd.read_excel('C:/Users/MBBLABS/Desktop/Text_sum/Input_LLM_summarization.xlsx')


# ## Basic Prompt

# ### Code Run

# In[ ]:


from langchain_community.document_loaders import DataFrameLoader
from langchain_community.chat_models import ChatOllama
from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage
)


# In[ ]:


# %%time
output_bp = []
time_lst_bp = []

for i in range(df.shape[0]):
# for i in range(0, 20):
    start_time = time.monotonic()
    
    ## load one article from the dataframe at a time
    df_1 = df.loc[[i]]
    df_1 = df_1[['Article_lat']]

    ## convert the dataframe with the loaded article into langchain document
    loader = DataFrameLoader(df_1, page_content_column="Article_lat")
    data = loader.load()

    # define messages to enhance the understanding of the summarization problem by the llm
    messages = [
        SystemMessage(content="You are an expert copywriter with expertise in summarizing documents."),
        HumanMessage(content=f'Write a summary including any related information about Name(s), Age(s), Date(s), Crime(s) committed, and their associated dates. Please provide a short and concise summary of the following text:\n TEXT:{data}')
    ]

    # initialize llm model object with the required model name and temperature
    llm = ChatOllama(model="llama3", temperature=0) # zero temperature to control hallucinations

    # invoke the llm model on the defined messages with 'data' as the input
    summary_output = llm.invoke(messages)

    
    end_time = time.monotonic()
    time_taken = end_time - start_time
    
    print(i)
    # append the output to the list
    output_bp.append(summary_output.content)
    time_lst_bp.append(time_taken)


# In[ ]:


# save the final output summary for the entire input sheet
out_df_bp = pd.DataFrame({'Article':df['Article_lat'].iloc[10:20].tolist(), 'Sum_Basic_Prompt':output_bp, 'Time_taken':time_lst_bp})
out_df_bp.to_csv('LLM_Output_BP.csv', index=False)


# ## Prompt Templates

# ### Code Run

# In[ ]:


from langchain_community.document_loaders import DataFrameLoader
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time
from datetime import timedelta


# In[ ]:


template = '''
Write a summary including any related information about Name(s), Age(s), Date(s), Crime(s) committed, and their associated dates. 
Please provide a short and concise summary of the following text:
TEXT:{data}
'''


# In[ ]:


# initialize llm model object with the required model name and temperature

# st_time = time.monotonic()
llm = ChatOllama(model="llama3", temperature=0)
# end_time = time.monotonic()
# elapsed_time = end_time - st_time
# print(elapsed_time)


# In[ ]:


# %%time
output_pt = []
time_lst_pt = []

for i in range(df.shape[0]):
# for i in range(15, 20):#, 20):
    start_time = time.monotonic()
    
    ## load one article from the dataframe at a time
    df_1 = df.loc[[i]]
    df_1 = df_1[['Article_lat']]

    ## convert the dataframe with the loaded article into langchain document
    loader = DataFrameLoader(df_1, page_content_column="Article_lat")
    data = loader.load()

    # define prompt template to enhance the understanding of the summarization problem by the llm
    prompt = PromptTemplate(
        input_variables=['data'],
        template=template
    )

    # initialize llm model object with the required model name and temperature
    # llm = ChatOllama(model="llama3", temperature=0) # zero temperature to control hallucinations

    # define the llm chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # invoke the llm chain on the defined messages with 'data' as the input
    summary_output = chain.invoke({'data':data})

    end_time = time.monotonic()
    time_taken = end_time - start_time

    print(i)
    # append the output to the list
    output_pt.append(summary_output['text'])
    time_lst_pt.append(time_taken)


# In[ ]:


# save the final output summary for the entire input sheet
out_df_pt = pd.DataFrame({'Article':df['Article_lat'].iloc[15:20].tolist(), 'Sum_Prompt_Templates':output_pt, 'Time_taken':time_lst_pt})
out_df_pt.to_csv('LLM_Output_PT.csv', index=False)


# ## Stuff Document Chain

# ### Code Run

# In[ ]:


from langchain_community.document_loaders import DataFrameLoader
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import time
from datetime import timedelta


# In[ ]:


template = '''
Write a summary including any related information about Name(s), Age(s), Date(s), Crime(s) committed, and their associated dates. 
Please provide a short and concise summary of the following text:
TEXT:{data}
'''


# In[ ]:


llm = ChatOllama(model="llama3", temperature=0)


# In[ ]:


# %%time
output_sdc = []
time_lst_sdc = []

for i in range(df.shape[0]):
# for i in range(0, 5):#, 20):
    start_time = time.monotonic()
    
    ## load one article from the dataframe at a time
    df_1 = df.loc[[i]]
    df_1 = df_1[['Article_lat']]

    ## convert the dataframe with the loaded article into langchain document
    loader = DataFrameLoader(df_1, page_content_column="Article_lat")
    data = loader.load()

    # define prompt template to enhance the understanding of the summarization problem by the llm
    prompt = PromptTemplate(
        input_variables=['data'],
        template=template
    )

    # initialize llm model object with the required model name and temperature
    # llm = ChatOllama(model="llama3", temperature=0) # zero temperature to control hallucinations

    # define the load_summarize_chain
    chain = load_summarize_chain(llm=llm, chain_type='stuff', prompt=prompt, document_variable_name="data", verbose=False)
    
    # invoke the llm chain on the defined messages with 'data' as the input
    summary_output = chain.invoke(data)

    end_time = time.monotonic()
    time_taken = end_time - start_time

    print(i)
    # append the output to the list
    output_sdc.append(summary_output['output_text'])
    time_lst_sdc.append(time_taken)


# In[ ]:


# save the final output summary for the entire input sheet
out_df_sdc = pd.DataFrame({'Article':df['Article_lat'].iloc[0:5].tolist(), 'Sum_Prompt_Templates':output_sdc, 'Time_taken':time_lst_sdc})
out_df_sdc.to_csv('LLM_Output_SDC.csv', index=False)

