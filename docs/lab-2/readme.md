# Retrieval Augmented Generation (RAG) with Langchain

[Retrieval Augumented Generation (RAG)](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) is an architectural pattern that can be used to augment the performance of language models by recalling factual information from a knowledge base, and adding that information to the model query.

The goal of this lab is to show how you can use RAG with an [IBM Granite](https://www.ibm.com/granite) model to augment the model query answer using a publicly available document. The most common approach in RAG is to create dense vector representations of the knowledge base in order to retrieve text chunks that are semantically similar to a given user query.

RAG use cases include:
- Customer service: Answering questions about a product or service using facts from the product documentation.
- Domain knowledge: Exploring a specialized domain (e.g., finance) using facts from papers or articles in the knowledge base.
- News chat: Chatting about current events by calling up relevant recent news articles.

In its simplest form, RAG requires 3 steps:

- Initial setup:
  - Index knowledge-base passages for efficient retrieval. In this recipe, we take embeddings of the passages and store them in a vector database.
- Upon each user query:
  - Retrieve relevant passages from the database. In this recipe, we use an embedding of the query to retrieve semantically similar passages.
  - Generate a response by feeding retrieved passage into a large language model, along with the user query.



## Prerequisites

This lab is a [Jupyter notebook](https://jupyter.org/). Please follow the instructions in [pre-work](../pre-work/readme.md) to run the lab.



## Loading the Lab

To run the notebook from your command line in Jupyter using the active virtual environment from the [pre-work](../pre-work/readme.md), run:

```shell
jupyter-lab
```

When Jupyter Lab opens the path to the `notebooks/RAG_with_Langchain.ipynb` notebook file is relative to the `sample-wids` folder from the git clone in the [pre-work](../pre-work/readme.md). The folder navigation pane on the left-hand side can be used to navigate to the file. Once the notebook has been found it can be double clicked and it will open to the pane on the right. 



## Running and Lab (with explanations)

This notebook demonstrates an application of long document summarisation techniques to a work of literature using Granite.

The notebook contains both `code` cells and `markdown` text cells. The text cells each give a brief overview of the code in the following code cell(s). These cells are not executable. You can execute the code cells by placing your cursor in the cell and then either hitting the **Run this cell** button at the top of the page or by pressing the `Shift` + `Enter` keys together. The main `code` cells are described in detail below.



## Choosing the Embeddings Model

```python
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
embeddings_model = HuggingFaceEmbeddings(
    model_name=embeddings_model_path,
)
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)
```

Here we are using the Hugging Face Transformers library to load a pre-trained model for generating embeddings (vector representations of text). Here's a breakdown of what each line does:

1. `from langchain_huggingface import HuggingFaceEmbeddings`: This line imports the `HuggingFaceEmbeddings` class from the 

   `langchain_huggingface` module. This class is used to load pre-trained models for generating embeddings.

2. `from transformers import AutoTokenizer`: This line imports the `AutoTokenizer` class from the `transformers` library. This class is used to tokenize text into smaller pieces (words, subwords, etc.) that can be processed by the model.

3. `embeddings_model_path = "ibm-granite/granite-embedding-30m-english"` : This line sets a variable `embeddings_model_path` to the path of the pre-trained model. In this case, it's a model called "granite-embedding-30m-english" developed by IBM's Granite project.

4. `embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_path)`: This line creates an instance of the `HuggingFaceEmbeddings` class, loading the pre-trained model specified by `embeddings_model_path`.

5. `embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)`: This line creates an instance of the `AutoTokenizer` class, loading the tokenizer that was trained alongside the specified model. This tokenizer will be used to convert text into a format that the model can process.

In summary, we are setting up a system for generating embeddings from text using a pre-trained model and its associated tokenizer. The embeddings can then be used for various natural language processing tasks, such as text classification, clustering, or similarity comparison.

To use a model from a provider other than Huggingface, replace this code cell with one from [this Embeddings Model recipe](https://github.com/ibm-granite-community/granite-kitchen/blob/main/recipes/Components/Langchain_Embeddings_Models.ipynb).



## Vector Database

```python
from langchain_milvus import Milvus
import tempfile

db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
print(f"The vector database will be saved to {db_file}")

vector_db = Milvus(
    embedding_function=embeddings_model,
    connection_args={"uri": db_file},
    auto_id=True,
    index_params={"index_type": "AUTOINDEX"},
)
```

This Python script is setting up a vector database using Milvus, a vector database built for AI applications, and Hugging Face's Transformers library for embeddings. It uses the previously created Embeddings Model. Here's a breakdown of what the code does:

1. It imports `tempfile` and `Milvus` from `langchain_milvus`.
2. It creates a temporary file for the Milvus database using `tempfile.NamedTemporaryFile()`. This file will store the vector database.
3. It initializes an instance of `Milvus`with the embedding function set to the previously created `embeddings_model`. The connection arguments specify the URI of the database file, which is the temporary file created in the previous step. The `auto_id` parameter is set to True, which means Milvus will automatically generate IDs for the vectors. The `index_params` parameter sets the index type to "AUTOINDEX", which allows Milvus to automatically choose the most suitable index for the data.

In summary, this script sets up a vector database using Milvus and a pre-trained embedding model from Hugging Face. The database is stored in a temporary file, and it's ready to index and search vector representations of text data.



## Selecting your model

Select a Granite model to use. Here we use a Langchain client to connect to  the model. If there is a locally accessible Ollama server, we use an  Ollama client to access the model. Otherwise, we use a Replicate client  to access the model.

When using Replicate, if the `REPLICATE_API_TOKEN` environment variable is not set, or a `REPLICATE_API_TOKEN` Colab secret is not set, then the notebook will ask for your [Replicate API token](https://replicate.com/account/api-tokens) in a dialog box.

```python
try:  # Look for a locally accessible Ollama server for the model
    response = requests.get(os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model = OllamaLLM(
        model="granite3.2:2b",
        num_ctx=65536,  # 64K context window
    )
    model = model.bind(raw=True)  # Client side controls prompt
except Exception:  # Use Replicate for the model
    model = Replicate(
        model="ibm-granite/granite-3.2-8b-instruct",
        replicate_api_token=get_env_var("REPLICATE_API_TOKEN"),
        model_kwargs={
            "max_tokens": 2000,  # Set the maximum number of tokens to generate as output.
            "min_tokens": 200,  # Set the minimum number of tokens to generate as output.
            "temperature": 0.75,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    )
```

In this first piece of code we **try** to determine if there is a local Ollama server running on `http://127.0.0.1:11434`. If the Ollama server is found then an `OllamaLLM` model instance is created for use later. If the Ollama server is not found the code then reverts to using the Granite 3.2-8b model served from Replicate .



# Split the document into chunks

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=embeddings_tokenizer,
    chunk_size=embeddings_tokenizer.max_len_single_sentence,
    chunk_overlap=0,
)
texts = text_splitter.split_documents(documents)
for doc_id, text in enumerate(texts):
    text.metadata["doc_id"] = doc_id
print(f"{len(texts)} text document chunks created")

```

This Python script is using the Langchain library to load a text file and split it into smaller chunks. Here's a breakdown of what each part does:

1. `from langchain.document_loaders import TextLoader`: This line imports the TextLoader class from the langchain.document_loaders module. TextLoader is used to load documents from a file.
2. `from langchain.text_splitter import CharacterTextSplitter` : This line imports the CharacterTextSplitter class from the `langchain.text_splitter` module. `CharacterTextSplitter` is used to split text into smaller chunks.
3. `loader = TextLoader(filename)` : This line creates an instance of `TextLoader`, which is used to load the text from the specified file `(filename)`.
4. `documents = loader.load()` : This line loads the text from the file and stores it in the `documents` variable as a list of strings.
5. `text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(...)` : This line creates an instance of `CharacterTextSplitter`. It takes a Hugging Face tokenizer `(embeddings_tokenizer)`, sets the chunk size to the maximum length of a single sentence that the tokenizer can handle, and sets the chunk overlap to 0 (meaning no overlap between chunks).
6. `texts = text_splitter.split_documents(documents)`: This line splits the documents into smaller chunks using the `CharacterTextSplitter` instance. The result is stored in the texts variable as a list of lists, where each inner list contains the chunks of a single document.
7. `for doc_id, text in enumerate(texts): text.metadata["doc_id"] = doc_id`: This loop assigns a unique identifier (doc_id) to each chunk of text. The doc_id is the index of the chunk in the texts list.
8. `print(f"{len(texts)} text document chunks created")`: This line prints the total number of text chunks created.

In summary, this script loads a text file, splits it into smaller chunks based on the maximum sentence length that a Hugging Face tokenizer can handle, assigns a unique identifier to each chunk, and then prints the total number of chunks created.



## Populate the vector database

```python
ids = vector_db.add_documents(texts)
print(f"{len(ids)} documents added to the vector database")
```

Next we load the `texts` object created earlier, split it into sentence-sized chunks, and adds these chunks to our vector database, associating each chunk with a unique ID.

1. `ids = vector_db.add_documents(texts)`: This line adds the text chunks to a vector database (

   `vector_db`). The `add_documents` method returns a list of IDs for the added documents.

2. `print(f"{len(ids)} documents added to the vector database")`: This line prints the number of documents added to the vector database.

   

## Querying the Vector Database



## Conduct a similarity search

Search the database for similar documents by proximity of the embedded vector in vector space.

```python
query = "What did the president say about Ketanji Brown Jackson?"
docs = vector_db.similarity_search(query)
print(f"{len(docs)} documents returned")
for doc in docs:
    print(doc)
    print("=" * 80)
```







## Answering Questions

Build a RAG chain with the model and the document retriever.

First we create the prompts for Granite to perform the RAG query. We use the Granite chat template and supply the placeholder values that the LangChain RAG pipeline will replace.

`{context}` will hold the retrieved chunks, as shown in the previous search, and feeds this to the model as document context for answering our question.

Next, we construct the RAG pipeline by using the Granite prompt templates previously created.



```python
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Create a Granite prompt for question-answering with the retrieved context
prompt = tokenizer.apply_chat_template(
    conversation=[{
        "role": "user",
        "content": "{input}",
    }],
    documents=[{
        "title": "placeholder",
        "text": "{context}",
    }],
    add_generation_prompt=True,
    tokenize=False,
)
prompt_template = PromptTemplate.from_template(template=prompt)

# Create a Granite document prompt template to wrap each retrieved document
document_prompt_template = PromptTemplate.from_template(template="""\
Document {doc_id}
{page_content}""")
document_separator="\n\n"

# Assemble the retrieval-augmented generation chain
combine_docs_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt_template,
    document_prompt=document_prompt_template,
    document_separator=document_separator,
)
rag_chain = create_retrieval_chain(
    retriever=vector_db.as_retriever(),
    combine_docs_chain=combine_docs_chain,
)
```



## Generate a retrieval-augmented response to a question

Use the RAG chain to process a question. The document chunks relevant to that question are retrieved and used as context.

```python
output = rag_chain.invoke({"input": query})

print(output['answer'])
```



## Credits

This notebook is a modified version of the IBM Granite Community [Retrieval Augmented Generation (RAG) with Langchain](https://github.com/ibm-granite-community/granite-snack-cookbook/blob/main/recipes/RAG/RAG_with_Langchain.ipynb) notebook. Refer to the [IBM Granite Community](https://github.com/ibm-granite-community) for the official notebooks.