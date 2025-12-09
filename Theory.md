**Gen AI** <br />
Text : transformer models
image: Generative Adversial Networks, wav2vec, diffusion models, VAE ( Variational Encoders) or CNNs. <br />
audio: GANs for audio, Deep generative models, RNN
video: GANs, CNN



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
        


        iii) prompt = PromptTemplate(
                                    template = """You are a classifier. Given a message, respond with a list of relevant labels.
                                                - Use **"idp"** for IDPs
                                                - Use **"smart"** for SMART goals or goals
                                                - Use **"performance"** for performance (PDR)
                                                - Use **"progress"** for course progression
                                                - Use **"hours"** for learning hours
                                                Only include labels that apply to the message. Return them as a JSON list under the `classification` key.
                                            
                                            Messages: "{message}

                                            {format_instructions}
                                        """,
                                        input_variables=["message"],
                                        partial_variables={"format_instructions": response_parser.get_format_instructions()}
                                    )
            
        formatted_prompt = prompt.format_prompt(message=state['history_plus_latest_message'],
                                                    )
        
        llm_response = llm(formatted_prompt.to_messages())
        
selected_llm=llm_response.response_metadata['model_name']
iv) prompt=ChatPromptTemplate.from_template("""
        - Respond with **"balance"** if user asks about their leave balance/remote working days balance. i.e. how many days of leaves/remote working days they have used and             how many left
        - Response with **"transaction"** if user asks about their leave transactions. i.e. how and when their leaves were used. e.g. sick leave on 10th March
        - Respond with both **"balance"** and **"transaction"**  if user asks for both
        Messages: "{message}"
        """)
    pr=prompt.format_prompt(message="")
    llm(pr.to_messages()
iv) chain = ChatPromptTemplate.from_template("""
        - Respond with **"balance"** if user asks about their leave balance/remote working days balance. i.e. how many days of leaves/remote working days they have used and             how many left
        - Response with **"transaction"** if user asks about their leave transactions. i.e. how and when their leaves were used. e.g. sick leave on 10th March
        - Respond with both **"balance"** and **"transaction"**  if user asks for both
        Messages: "{message}"
        """) | llm

        users = check_nminus1(state)["employees"]
        log_execution(state['thread_id'], "Node", "get_leave_details", f"[The return value of user from check_nminus1 is : \n{users}]")
        

        response = chain.invoke({"message": state['history_plus_latest_message']})
        
        selected_llm=response.response_metadata['model_name']

 prompt = ChatPromptTemplate.from_messages([
            ("system", rectify_hallucination_toxicity_prompt),
        ])

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
A LangChain agent is like an LLM-powered decision-maker that allows llm to decide which tools to use and take actions in structured manner to solve a task.
Mostly we use React agents ( Reasoning + Analysis ).React phenomena works in manner, based on the input query to LLm <br/> reasoning happens and accordingly actions is decided along with the tool for it.Now these observations are fed to LLM to <br/> generate appropriate output.Now to create React agent we need to have prompt composed of <br/> Thought,Action,Pause,Observation,Output. Hence we create prompt that is composed of sections and initialize  agent using <br/> this prompt and then pass the query. We get output from agent as the action and the thought in messages. Now based on <br/> action tool is decided (def as function) and output of tool again to LLM as observations and finally generate output. <br/> This iterative process is better orchestrated with Langraph. refer langraph section <br/>
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

LangGraph supports persisted state using built-in storage so that long-running or multi-session agents can pause/resume.

How persistence works

Each graph run is tracked through a checkpoints or threads directory.State for each step is stored (usually as JSON) including:messages,node outputs,metadata,partial results
You can resume execution from any checkpoint.

LangGraph natively supports human approval or intervention within the graph flow.

How HITL Works in LangGraph

A node can return:

{"await_user_input": True}

This causes LangGraph to pause execution and waits for external input.

You can resume by sending the user’s message back to the graph using an API call.

Typical HITL patterns:

“Approval required” step
“Provide missing details” workflow
Human validation before publishing
Human correction loop (like review/edit cycles)


**Components of graph** 
node,edges and conditional edges
There exist agent state accessible for all parts of graph.It is local to ;angraph and the simplest way to define agent state is : <br/>
class AgentState(TypedDict): <br/>
    messages: Annotated[list[AnyMessage], operator.add] <br/>
Now here we have message as variable which is anymessage (a langchain type), on that we have operator add that means to append as list of messages as we move head in graph.


 

https://langchain-ai.github.io/langgraph/ <br/>


**Multi-modal RAG**
https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb?ref=blog.langchain.dev <br/>
How to extract multi modal contemt from PDF - pyMuPDF4LLM API <br/>
https://thetredenceacademy.talentlms.com/unit/view/id:18063
How to extract multi modal contemt from DOC <br/>
https://thetredenceacademy.talentlms.com/unit/view/id:18064
How to extract multi modal contemt from PPT <br/>
https://python-pptx.readthedocs.io/en/latest/index.html#api-documentation <br/>

**Statistical metrics**
Staistical Metrics - BLEU, ROUGHE, METEOR
PARENT - Measures using both target & source text 
BVSS - Bag of vectors sentence similarity

**Model-based metrics**
1) IE - evaluation based on info extratced
2) QA - measures similarity b/w questions answered
3) Faithfulness - an unsupported info in the output
4) LM based calcultes ratio of hallucinated tokens & total tokens

   
**How to handle model hallucinations **
Use Retrieval-Augmented Generation (RAG)
Combine LLMs with external knowledge bases or search APIs.

Example tools: LangChain, Haystack.

The model pulls context from a trusted source at runtime to ground its response.

✅ Output is “grounded” in real data.

B. Fine-tuning or Instruction Tuning
Fine-tune the model on domain-specific, verified data.

Use instruction tuning to teach the model how to respond cautiously.

C. Prompt Engineering
Use clear, constrained prompts to reduce hallucination.

Add guardrails:

text
Copy
Edit
If you don’t know the answer, say “I don’t know.”
Only respond based on the context provided below:
Use few-shot examples that demonstrate the desired behavior.

D. Model Choice
Use models known for higher factual accuracy (e.g., GPT-4-turbo).

Avoid older or smaller models for tasks requiring high reliability.

🧪 3. Post-processing / Validation
Fact-checking APIs: Run outputs through fact-checking pipelines.

Confidence scoring: Some setups can provide confidence in the output (RAG models with passage scores).

Structured output formats: Use formats like JSON to constrain randomness.







