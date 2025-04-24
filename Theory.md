**Prompt Engineering** <br />

1)systemprompt = 
    '''
    You are a data scientist working for a company to create graph database. -------------------  [ AI definition ] 
    Your task is to extract information and convert it into knowledge graphs. ------------------- -------[ Objective ] 
    Provide a set of nodes in form of [head, head_type,tail, tail_type,relation] .-------------   [ Definition ] 
    It is necessary that head and tail exist as nodes that are related by relation.If can't pair a relatiosnhip b/w nodes then please don't     
    provide it.
    When you find a node or relationship you want to add try to create a generic TYPE for it that describes the entity you can also think of it 
    as a label.
    You must generate the output in a JSON format containing a list with JSON objects. Each object should have the keys: "head", "head_type", 
    "relation", "tail", and "tail_type". -----  [ Output format ] 
    '''


2)Human Prompt 
Human prompt contains some instructions / tip or important note and input variable.
  1)
  2) human_prompt = PromptTemplate(
            template="""
        Examples:
        {examples}
        
        For the following text, extract entities and relations as in the provided example.
        {format_instructions}\nText: {input}""",
            input_variables=["input"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "node_labels": None,
                "rel_types": None,
                "examples": examples,
            },
)
          human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)    <br />
          Refer this link https://api.python.langchain.com/en/latest/_modules/langchain_experimental/graph_transformers/llm.html <br />
3) Chat Prompt
          i)  default_prompt = ChatPromptTemplate.from_messages(
                [
                        (
                            "system",
                            system_prompt,
                        ),
                        (
                            "human",
                            (
                                "Tip: Make sure to answer in the correct format and do "
                                "not include any explanations. "
                                "Use the given format to extract information from the "
                                "following input: {input}"
                            ),
                        ),
                    ]
                )

         ii) prompt_template_entity = ChatPromptTemplate.from_messages([ ("system", f'''{system_prompt}'''), ("human", "{user_input}")])
**RAG**  <br />
https://dheerajinampudi.medium.com/retrieval-chains-enhancing-rags-with-different-retrieval-techniques-c6071f1a0ff3


**Few shot learning** 
<br />
https://python.langchain.com/v0.1/docs/modules/model_io/prompts/few_shot_examples_chat/ <br />

**ChatPromptTemplate**  <br />
https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html#langchain_core.prompts.chat.ChatPromptTemplate <br />

**Sequential Chains**  <br />

https://www.analyticsvidhya.com/blog/2023/10/a-comprehensive-guide-to-using-chains-in-langchain/
https://www.analyticsvidhya.com/blog/2023/12/implement-huggingface-models-using-langchain/
https://www.comet.com/site/blog/chaining-the-future-an-in-depth-dive-into-langchain/
https://github.com/langchain-ai/langchain/discussions/16421




**Output Parser**
The basic implementation to introduce the output parser in langchain is presented here :- <br />
https://medium.com/@larry_nguyen/langchain-101-lesson-3-output-parser-406591b094d7 <br />
https://python.plainenglish.io/langchain-in-chains-7-output-parsers-e1a2cdd40cd3 <br />

**1) Json OutputParser**
https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/json/

**2) StructuredOutputParser** <br />
https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/json/  < br />
https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/structured/

**3) Pydantic output parser** <br />
https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/pydantic/  <br />
Takes a user defined Pydantic model and returns data in that format.\


**Applications** <br/>
1)In NER & Relationship extraction using graph transformers and LLMchain <br />
https://github.com/Siddharthg97/NER <br />
2)Summarization <br />
3) Scraping <br />
refer scraping repository
4) Functional agents


-------------------------------------------------------------------------------------------------------------------------------------------------------------

**Memory storage in Langchains**  <br/>
ConversationBufferMemory
ConversationBufferWindowMemory
ConversationTokenBufferMemory
ConversationSummaryMemory


**Langchain Agent** <br/>
Mostly we use React agents ( Reasoning + Analysis ).React phenomena works in manner, based on the input query to LLm  reasoning happens and accordingly actions is decided along with the tool for it.Now these observations are fed to LLM to generate appropriate output.Now to create React agent we need to have prompt composed of Thought,Action,Pause,Observation,Output. Hence we create prompt that is composed of sections and initialize  agent using this prompt and then pass the query. We get output from agent as the action and the thought in messages. Now based on action tool is decided (def as function) and output of tool again to LLM as observations and finally generate output. This iterative process is better orchestrated with Langraph. refer langraph section <br/>
refer React Agents.ipynb
https://python.langchain.com/docs/concepts/agents/ <br/>

https://python.langchain.com/v0.1/docs/modules/agents/how_to/custom_agent/ <br />
https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/ <br/>

**Langraph**
LangGraph is an extension of LangChain specifically that support graphs, aimed at creating highly controllable and customizable agents. We recommend that you use LangGraph for building agents. Single and multi-agent flows are desrcibed and represented as graph. Allows for extremely  controlled flows, build it persistance for human-in-loop workflows. <br/>
We can create 
1) cyclic graph
2) persistance
3) human-in-loop

Components of graph 
node,edges and conditional edges
There exist agent state accessible for all parts of graph

 

https://langchain-ai.github.io/langgraph/ <br/>


**Multi-modal RAG**
https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb?ref=blog.langchain.dev <br/>
How to extract multi modal contemt from PDF - pyMuPDF4LLM API <br/>
https://thetredenceacademy.talentlms.com/unit/view/id:18063
How to extract multi modal contemt from DOC <br/>
https://thetredenceacademy.talentlms.com/unit/view/id:18064
How to extract multi modal contemt from PPT <br/>
https://python-pptx.readthedocs.io/en/latest/index.html#api-documentation <br/>



**How d **








