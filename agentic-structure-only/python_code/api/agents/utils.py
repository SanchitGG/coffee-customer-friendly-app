
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("GROQ_API_KEY")
def get_chatbot_response(messages, model_name="llama-3.1-8b-instant",  temperature=0.7):
    
    client = Groq()

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        top_p=0.8,
        max_tokens=2048,
    )

    return response.choices[0].message.content.strip()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
def get_embedding(text_input):
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type= "retrieval_query"
    )
    vector = embeddings.embed_query(text_input)


    return vector


def double_check_json_output(json_string):
    prompt = f""" You will check this json string and correct any mistakes that will make it invalid. Then you will return the corrected json string. Nothing else. 
    If the Json is correct just return it.

    Do NOT return a single letter outside of the json string.

    {json_string}
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_chatbot_response(messages)

    return response