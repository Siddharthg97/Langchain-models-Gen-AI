**Prompt Engineering** <br />
2)systemprompt = 
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


1)Human Prompt 
There are many ways to create the human prompt
  1) prompt_template_entity = ChatPromptTemplate.from_messages([ ("system", f'''{system_prompt}'''), ("human", "{user_input}"),])
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
          human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

**RAG** <br />
https://dheerajinampudi.medium.com/retrieval-chains-enhancing-rags-with-different-retrieval-techniques-c6071f1a0ff3


**Few shot learning** < br />
https://python.langchain.com/v0.1/docs/modules/model_io/prompts/few_shot_examples_chat/ <br />


**Sequential Chains**

https://www.analyticsvidhya.com/blog/2023/10/a-comprehensive-guide-to-using-chains-in-langchain/
https://www.analyticsvidhya.com/blog/2023/12/implement-huggingface-models-using-langchain/
https://www.comet.com/site/blog/chaining-the-future-an-in-depth-dive-into-langchain/
https://github.com/langchain-ai/langchain/discussions/16421




**Output Parser**

https://python.plainenglish.io/langchain-in-chains-7-output-parsers-e1a2cdd40cd3 <br />

**1) Json OutputParser**
https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/json/

**2) StructuredOutputParser** <br />
https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/json/  < br />
https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/structured/

**3) Pydantic output parser** <br />
https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/pydantic/  <br />
Takes a user defined Pydantic model and returns data in that format.\


**Applications**
1)In NER & Relationship extraction using graph transformers and LLMchain <br />
https://github.com/Siddharthg97/NER

