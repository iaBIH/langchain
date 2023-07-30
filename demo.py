# source: 
# https://github.com/gkamradt/langchain-tutorials/blob/main/LangChain%20Cookbook%20Part%202%20-%20Use%20Cases.ipynb
# https://github.com/langchain-ai/langchain/blob/490ad93b3cf7d24b30f8993f860b654ff107e638/docs/extras/integrations/toolkits/pandas.ipynb#L8
# pip install langchain openai  tiktoken faiss-cpu faiss-gpu tabulate
#

from dotenv import load_dotenv
import os
import pandas as pd

from langchain.llms import OpenAI 
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# The vectorstore we'll be using
from langchain.vectorstores import FAISS
# The LangChain component we'll use to get the documents
from langchain.chains import RetrievalQA
# The easy document loader for text
from langchain.document_loaders import TextLoader
# The embedding engine that will convert our text to vectors
from langchain.embeddings.openai import OpenAIEmbeddings
# To help construct our Chat Messages
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
# We will be using a chat model, defaults to gpt-3.5-turbo
from langchain.chat_models import ChatOpenAI
# To parse outputs and get structured data back
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# Text splitters
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

api_key = ''

#************************* 1. Summarization 
def example_1_1():
    print("=========== Summaries Of Short Text ==================")

    load_dotenv()

    openai_api_key = os.getenv('OPENAI_API_KEY', api_key)

    # Note, the default model is already 'text-davinci-003' but I call it out here explicitly so you know where to change it later if you want
    llm = OpenAI(temperature=0, model_name='text-davinci-003', openai_api_key=api_key)

    # Create our template
    template = """
    %INSTRUCTIONS:
    Please summarize the following piece of text.
    Respond in a manner that a 5 year old would understand.

    %TEXT:
    {text}
    """

    # Create a LangChain prompt template that we can insert values to later
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )

    confusing_text = """
    For the next 130 years, debate raged.
    Some scientists called Prototaxites a lichen, others a fungus, and still others clung to the notion that it was some kind of tree.
    “The problem is that when you look up close at the anatomy, it’s evocative of a lot of different things, but it’s diagnostic of nothing,” says Boyce, an associate professor in geophysical sciences and the Committee on Evolutionary Biology.
    “And it’s so damn big that when whenever someone says it’s something, everyone else’s hackles get up: ‘How could you have a lichen 20 feet tall?’”
    """

    print ("------- Prompt Begin -------")

    final_prompt = prompt.format(text=confusing_text)
    print(final_prompt)

    print ("------- Prompt End -------")

    output = llm(final_prompt)
    print (output)


def example_1_2():
    print("=========== Summaries Of Longer Text ==================")
    llm = OpenAI(temperature=0, openai_api_key=api_key)
    with open('sampleText1.txt', 'r') as file:
         text = file.read()
    # Printing the first 285 characters as a preview
    print (text[:285])

    num_tokens = llm.get_num_tokens(text)
    print (f"There are {num_tokens} tokens in your file")
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=350)
    docs = text_splitter.create_documents([text])

    print (f"You now have {len(docs)} docs intead of 1 piece of text")
    # Get your chain ready to use
    chain = load_summarize_chain(llm=llm, chain_type='map_reduce') # verbose=True optional to see what is getting sent to the LLM
    # Use it. This will run through the 4 documents, summarize the chunks, then get a summary of the summary.
    output = chain.run(docs)
    print (output)

#************************* 2. Question and Answer 
def example2_1():
    print("=========== Q A Example ==================")
    llm = OpenAI(temperature=0, openai_api_key=api_key)
    context = """
    Rachel is 30 years old
    Bob is 45 years old
    Kevin is 65 years old
    """
    question = "Who is under 40 years old?"
    output = llm(context + question)
    # I strip the text to remove the leading and trailing whitespace
    print (output.strip())

    question = "Who is the oldest?"
    output = llm(context + question)
    # I strip the text to remove the leading and trailing whitespace
    print (output.strip())

def example2_2():
    print("=========== Using Embeddings ==================")
    llm = OpenAI(temperature=0, openai_api_key=api_key)
    loader = TextLoader('sampleText2.txt')
    doc = loader.load()
    print (f"You have {len(doc)} document")
    print (f"You have {len(doc[0].page_content)} characters in that document")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
    docs = text_splitter.split_documents(doc)

    #Get the total number of characters so we can see the average later
    num_total_characters = sum([len(x.page_content) for x in docs])

    print (f"Now you have {len(docs)} documents that have an average of {num_total_characters / len(docs):,.0f} characters (smaller pieces)")
    # Get your embeddings engine ready
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
    docsearch = FAISS.from_documents(docs, embeddings)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

    query = "What is the rare disease that affects most people in USA?"
    a = qa.run(query)
    print(a)
    query = "What is the rare disease that most deadly?"
    a = qa.run(query)
    print(a)

    query = "List all errors in the MDcharts.py"
    a = qa.run(query)
    print(a)

#************************* 3. Extraction
def example_3_1():
    print("=========== Extraction: Vanilla Extraction ==================")
    #llm = OpenAI(temperature=0, openai_api_key=api_key)
    #loader = TextLoader('sampleText2.txt')
    chat_model = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=api_key)

    instructions = """
    You will be given a sentence with fruit names, extract those fruit names and assign an emoji to them
    Return the fruit name and emojis in a python dictionary
    """

    fruit_names = """
    Apple, Pear, this is an kiwi
    """
    # Make your prompt which combines the instructions w/ the fruit names
    prompt = (instructions + fruit_names)

    # Call the LLM

    output = chat_model([HumanMessage(content=prompt)])
    print (output.content)
    print (type(output.content))

    output_dict = eval(output.content)
    print (output_dict)
    print (type(output_dict))

def example_3_2():
    print("=========== Extraction: Using LangChain's Response Schema ==================")
    chat_model = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=api_key)

    # The schema I want out
    response_schemas = [
        ResponseSchema(name="artist", description="The name of the musical artist"),
        ResponseSchema(name="song"  , description="The name of the song that the artist plays")
    ]

    # The parser that will look for the LLM output in my schema and return it back to me
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    #The format instructions that LangChain makes. Let's look at them
    format_instructions = output_parser.get_format_instructions()
    print("format_instructions: ", format_instructions)
    print("------------------------")
    # The prompt template that brings it all together
    # Note: This is a different prompt template than before because we are using a Chat Model
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("Given a command from the user,\
                                                      extract the artist and song names \n \
                                                        {format_instructions}\n{user_prompt}")  
        ],
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": format_instructions}
    )
    fruit_query = prompt.format_prompt(user_prompt="I really like So Young by Portugal. The Man")
    print (fruit_query.messages[0].content)
    fruit_output = chat_model(fruit_query.to_messages())
    output = output_parser.parse(fruit_output.content)
    print (output)
    print (type(output))

#************************* 4. Evaluation

#************************* 5. Query tabular data 

#************************* 6. Code understanding 
def example_6_1():
    print("=========== Extraction: Using LangChain's Response Schema ==================")
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=api_key)
    embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=api_key)    
    #loader = TextLoader('sampleText2.txt')
    root_dir = 'synth-md'
    docs = []
    # Go through each folder
    counter = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):        
        # Go through each file
        for fnm in filenames:
            #print(fnm)
            try: 
                # Load up the file as a doc and split
                counter += 1
                filePath  = os.path.join(dirpath, fnm)
                #print(filePath)
                loader = TextLoader(filePath, encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e: 
                pass
    print (f"You have {counter} files\n")
    print (f"You have {len(docs)} documents\n")
    print ("------ Start Document ------")
    print (docs[0].page_content[:300])
    print("------------  Code Analysis ------------------")
    docsearch = FAISS.from_documents(docs, embeddings)
    # Get our retriever ready
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    
    query = "Convert synth-md python package to maven java package"
    output = qa.run(query)
    print(output)


#************************* 7. Interaction with APIs

#************************* 8. Chatbots

#************************* 9. Agents

#************************* 10. FIN

#************************* 11. Generalization of attribute values 

def example_11_1():
    print("=========== Extraction: Generalization of attribute values ==================")
    dataPath = "data/adult500.csv"
    df = pd.read_csv(dataPath)
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0, openai_api_key=api_key), df, verbose=True)
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=api_key),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    print("------------------------")
    agent.run("how many rows are there?")
    # print("------------------------")
    # agent.run("whats the square root of the average age?")
    print("------------------------")
    agent.run("What are the unique values of workclass attribute?")
    print("------------------------")
    agent.run("Suggest 3 levels of generalizations for the values of workclass\
               attribute that can be used for k-anonymity?\
               format the output as json")
    print("------------------------")

    # df1 = df.copy()
    # df1["Age"] = df1["Age"].fillna(df1["Age"].mean())
    # agent = create_pandas_dataframe_agent(OpenAI(temperature=0), [df, df1], verbose=True)
    # agent.run("how many rows in the age column are different?")

print("xxxxxxxxxxxxxxxx LangChain Demo xxxxxxxxxxxxxxxx")

#example_1_1()
#example_1_2()
#example_2_1()
#example_2_1()
#example_3_1()
#example_3_2()
#example_6_1()
example_11_1()