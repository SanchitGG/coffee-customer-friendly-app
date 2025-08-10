import os
import json
from .utils import get_chatbot_response, double_check_json_output

from copy import deepcopy
from dotenv import load_dotenv
load_dotenv()


class OrderTakingAgent():
    def __init__(self, recommendation_agent):
        self.recommendation_agent = recommendation_agent
    
    def get_response(self, messages):
        messages = deepcopy(messages)
        system_prompt = """
            You are a customer support Bot for a coffee shop called "Merry's Way"

            Here is the menu for this coffee shop:

            Cappuccino - $4.50
            Jumbo Savory Scone - $3.25
            Latte - $4.75
            Chocolate Chip Biscotti - $2.50
            Espresso shot - $2.00
            Hazelnut Biscotti - $2.75
            Chocolate Croissant - $3.75
            Dark chocolate (Drinking Chocolate) - $5.00
            Cranberry Scone - $3.50
            Croissant - $3.25
            Almond Croissant - $4.00
            Ginger Biscotti - $2.50
            Oatmeal Scone - $3.25
            Ginger Scone - $3.50
            Chocolate syrup - $1.50
            Hazelnut syrup - $1.50
            Caramel syrup - $1.50
            Sugar Free Vanilla syrup - $1.50
            Dark chocolate (Packaged Chocolate) - $3.00

            Things to NOT DO:
            * DON'T ask how to pay by cash or Card.
            * Don't tell the user to go to the counter
            * Don't tell the user to go to a place to get the order

            Your task is as follows:
            1. Take the User's Order
            2. Validate that all their items are in the menu
            3. If an item is not in the menu, let the user know and repeat back the remaining valid order
            4. Ask them if they need anything else
            5. If they do, then repeat starting from step 3
            6. If they don't want anything else, using the "order" object that is in the output, make sure to hit all three points:
                1. List down all the items and their prices
                2. Calculate the total
                3. Thank the user for the order and close the conversation with no more questions

            Produce the following output without any additions, not a single letter outside of the structure below.
            Your output should be in a structured JSON format like so. Each key is a string and each value is a string. Make sure to follow the format exactly:
            {
                "chain of thought": "Write down your critical thinking about what is the maximum task number the user is on right now. Then write down your critical thinking about the user input and its relation to the coffee shop process. Then write down your thinking about how you should respond in the response parameter taking into consideration the Things to NOT DO section and focus on the things that you should not do.",
                "step number": "Determine which task you are on based on the conversation.",
                "order": "This is going to be a list of JSONs like so: [{'item': 'put the item name', 'quantity': 'put the number that the user wants from this item', 'price': 'put the total price of the item'}]",
                "response": "Write a response to the user"
            }
        """

        last_order_taking_status = ""
        content_dict = {}
        
        for message_index in range(len(messages) - 1, -1, -1):  # Fixed range to include index 0
            message = messages[message_index]
            
            agent_name = message.get("name", "")
            
            if message["role"] == "assistant" and agent_name == "order_taking_agent":
                try:
                    content_dict = json.loads(message["content"]) if isinstance(message["content"], str) else message["content"]
                    step_number = int(content_dict["step number"])
                    order = content_dict["order"]
                   
                    last_order_taking_status = f"""
                    step number: {step_number}
                    order: {order}
                    """
                    break
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    # Handle cases where message content is not valid JSON or missing keys
                    continue
        
        # Handle the case where we have a last user message
        if messages and len(messages) > 0:
            if last_order_taking_status:
                # Get the response from content_dict if available
                last_response = content_dict.get('response', '') if content_dict else ''
                messages[-1]['content'] = last_order_taking_status + "\n" + last_response
            # If no previous order status, keep the original user message

        input_messages = [{"role": "system", "content": system_prompt}] + messages        

        chatbot_output = get_chatbot_response(input_messages)

        # Double check JSON 
        chatbot_output = double_check_json_output(chatbot_output)

        output = self.postprocess(chatbot_output, messages)

        return output

    def postprocess(self, output, messages):
        try:
            output = json.loads(output) if isinstance(output, str) else output
        except json.JSONDecodeError:
            # Handle invalid JSON gracefully
            return {
                "role": "assistant",
                "name": "order_taking_agent",
                "content": json.dumps({
                    "chain of thought": "Error parsing output",
                    "step number": "1",
                    "order": "[]",
                    "response": "I apologize, there was an error processing your request. Could you please repeat your order?"
                })
            }

        # Handle string representation of order list
        if isinstance(output["order"], str):
            try:
                output["order"] = json.loads(output["order"])
            except json.JSONDecodeError:
                output["order"] = []

        response = json.dumps(output)  # Convert back to JSON string for consistency
        
        # Check if order has fewer than 3 items and get recommendations
        if 0 < len(output["order"]) < 3:
            try:
                recommendation_output = self.recommendation_agent.get_recommendations_from_order(messages, output['order'])
                response = recommendation_output['content']
            except Exception as e:
                # Handle recommendation service failure gracefully
                pass

        dict_output = {
            "role": "assistant",
            "name": "order_taking_agent",
            "content": response,
        }

        return dict_output