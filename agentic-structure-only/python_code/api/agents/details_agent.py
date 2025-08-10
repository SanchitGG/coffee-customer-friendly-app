from dotenv import load_dotenv
import os
from .utils import get_chatbot_response,get_embedding

from copy import deepcopy
from pinecone import Pinecone
load_dotenv()

class DetailsAgent():
    def __init__(self):
        
       
        
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
    
    def get_closest_results(self,index_name,input_embeddings,top_k=2):
        index = self.pc.Index(index_name)
        
        results = index.query(
            namespace="ns1",
            vector=input_embeddings,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )

        return results

    def get_response(self,messages):
        messages = deepcopy(messages)

        user_message = messages[-1]['content']
        embedding = get_embedding(user_message)
        result = self.get_closest_results(self.index_name,embedding)
        source_knowledge = "\n".join([x['metadata']['text'].strip()+'\n' for x in result['matches'] ])

        prompt = f"""
        
        Using the contexts below, answer the query.

        Contexts:
        {source_knowledge}

        Query: {user_message}
        """

        system_prompt = """ You are a customer support agent for a coffee shop called Merry's way. You should answer every question as if you are waiter and provide the neccessary information to the user regarding their orders """
        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output =get_chatbot_response(input_messages)
        output = self.postprocess(chatbot_output)
        return output

    def postprocess(self,output):
        output = {
            "role": "assistant",
            "name": "details_agent",
            "content": output
            
        }
        return output

    
