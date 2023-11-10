import chainlit as cl
import torch
# from torch import cuda, bfloat16
# import transformers
from langchain.vectorstores import FAISS
from auto_gptq import AutoGPTQForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, TextStreamer, pipeline
# import chainlit as cl
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DB_FAISS_PATH = "/content/handbook-bge-embeddings/text_files/700-15"
embeddings_df = pd.read_parquet("/content/handbook-bge-embeddings/manual_embeddings.parquet")

model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_basename = "model"

embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5',
                                       model_kwargs={'device': 'cuda'})


my_template_3 = """[INST] You are a helpful bot that provides information related to user's question or topic of interest based on provided context.
----------------------------------------------------------------------
Here is provided text(context) and Question/topic:

Context: {context}
Question/topic: {question}
----------------------------------------------------------------------

Answer:
[/INST]"""

def set_custom_prompt():

    prompt = PromptTemplate(template=my_template_3, input_variables=['context', 'question'])

    return prompt


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        revision="main",
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        inject_fused_attention=False,
        device=DEVICE,
        quantize_config=None,
    )

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.01,
        # top_p=0.95,
        top_k=1,
        repetition_penalty=1,
        streamer=streamer
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.01, "stream": True})

    return llm


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 8}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}

    )

    return qa_chain

# ,'score_threshold': 0.7
@cl.cache
def qa_bot():
    # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    #                                    model_kwargs={'device': 'cuda'})
    # embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5',
    #                                    model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


def app_cvb(emb,query):


  cosine_similarity = 1 - cosine(emb, query)

  return cosine_similarity

def is_relavant(query):
  #res = embeddings_df['embedding'].apply(app_cvb, additional_arg=query)
  res = embeddings_df['embedding'].apply(lambda x: app_cvb(x, query))

  res = np.array(res)
  if np.sort(res)[-1] >= 0.6:
    print(np.sort(res)[-1])
    return True
  else:
    return False


@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="starting the bot....")
    await msg.send()
    msg.content = "Hello, Ask the question"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    aa_emb = embeddings.embed_query(message)
    if is_relavant(aa_emb):
      res = await chain.acall(message, callbacks=[cb])
      answer = res["result"]
    else:
      answer = """| Qualifying Degree      | Minimum Credits Through Coursework | Minimum Credits Through Research | Minimum Credits for Graduation | Minimum Duration to Graduate | Maximum Duration to Graduate |
|------------------------|-----------------------------------|----------------------------------|-------------------------------|-----------------------------|-------------------------------|
| Group A                | B.Tech., B.E. or equivalent 4-year bachelor's degree | 24 | 54 | 78 | 3.5 years | 7 years |
| Group B                | M.Sc., M.A. or equivalent | 18 | 54 | 72 | 3 years | 6 years |
| Group C                | M.Tech., M.E., or M.Phil. or equivalent | 9 | 54 | 63 | 2.5 years | 6 years |
"""
    # sources = res["source_documents"]
    #
    # if sources:
    #     answer += f"\nSources:" + str(sources)
    # else:
    #     answer += f"\nNo Sources Found"

    await cl.Message(content=answer).send()
