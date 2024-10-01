import json
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import time
from openai_utils import call_gpt, call_llama

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(key, messages, functions=None,function_call=None,key_pos=None, model="gpt-3.5-turbo-16k-0613", use_llama = False, stop=None,process_id=0, **args):
    use_messages = []
    for message in messages:
        if not("valid" in message.keys() and message["valid"] == False):
            use_messages.append(message)
    json_data = {
        "model": model,
        "messages": use_messages,
        "max_tokens": 1024,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        **args
    }
    if stop is not None:
        json_data.update({"stop": stop})
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        if not use_llama:
            openai_response = call_gpt(
                **json_data,
            )
        else:
            openai_response = call_llama(
                **json_data,
            )
        json_data = json.loads(openai_response.json())
        json_data["choices"][0]['message'].pop('tool_calls')
        return json_data 

    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"OpenAI calling Exception: {e}")
        return e
class ChatGPTFunction:
    def __init__(self, model="gpt-3.5-turbo-16k-0613", openai_key=""):
        self.model = model
        self.conversation_history = []
        self.openai_key = openai_key
        self.time = time.time()
        self.TRY_TIME = 6

    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def parse(self,functions,process_id,key_pos=None,**args):
        self.time = time.time()
        conversation_history = self.conversation_history
        for _ in range(self.TRY_TIME):
            if _ != 0:
                time.sleep(15)
            if functions != []:
                json_data = chat_completion_request(
                    self.openai_key, conversation_history, functions=functions,process_id=process_id, key_pos=key_pos, **args
                )                    
            else:
                json_data = chat_completion_request(
                    self.openai_key, conversation_history,process_id=process_id,key_pos=key_pos, **args
                )
            try:
                total_tokens = json_data['usage']['total_tokens']
                completion_tokens = json_data['usage']['completion_tokens']
                prompt_tokens = json_data['usage']['prompt_tokens']
                message = json_data["choices"][0]["message"]
                if process_id == 0:
                    print(f"[process({process_id})]total tokens: {json_data['usage']['total_tokens']}")

                if "function_call" in message.keys() and "." in message["function_call"]["name"]:
                    message["function_call"]["name"] = message["function_call"]["name"].split(".")[-1]

                return message, 0, total_tokens, completion_tokens, prompt_tokens
            except BaseException as e:
                print(f"[process({process_id})]Parsing Exception: {repr(e)}. Try again.")
                if json_data is not None:
                    print(f"[process({process_id})]OpenAI return: {json_data}")
            
class GPT4Function:
    def __init__(self, model="gpt-4-0613", openai_key=""):
        self.model = model
        self.conversation_history = []
        self.openai_key = openai_key
        self.time = time.time()
        self.TRY_TIME = 6

    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def filter_functions(self, functions, conversation_history):
        selected_functions_names = []
        for msg in conversation_history:
            if msg["role"] == "assistant" and "function_call" in msg.keys() and msg["function_call"] != None and msg["function_call"]["name"]=="function_selection":
                function2exection = msg["function_call"]["arguments"]
                json_data = json.loads(function2exection,strict=False)
                if "function_name" in json_data.keys():
                    function2exection = json_data["function_name"]
                    selected_functions_names.append(function2exection)
        final_result = []
        added_names = []
        for fun in functions:
            if fun["name"] in ["Finish", "function_selection"]:
                final_result.append(fun)
            if fun["name"] in selected_functions_names and fun["name"] not in added_names:
                added_names.append(fun["name"])
                final_result.append(fun)
        
        return final_result
    
        
    def parse(self,functions,process_id,key_pos=None, is_llama =False, **args):
        self.time = time.time()
        for conversation in self.conversation_history:
            if 'content' not in conversation:
                conversation['content'] = ''
                print(self.conversation_history, file=open('tmp.txt','a'))
        conversation_history = self.conversation_history

        # functions = self.filter_functions(functions, conversation_history) # TODO
        print("---------call-----------")
        print(conversation_history)
        print("---------functions-----------")
        print(functions)
        for _ in range(self.TRY_TIME):
            if _ != 0:
                time.sleep(15)
            if functions != []:
                if is_llama and len(functions) == 2:
                    json_data = chat_completion_request(
                        self.openai_key, conversation_history, functions=functions,process_id=process_id, key_pos=key_pos, model=self.model, use_llama = True, **args
                    )
                else:
                    json_data = chat_completion_request(
                        self.openai_key, conversation_history, functions=functions,process_id=process_id, key_pos=key_pos, model=self.model, **args
                    )
            else:
                json_data = chat_completion_request(
                    self.openai_key, conversation_history,process_id=process_id,key_pos=key_pos, model=self.model, **args
                )
            try:
                total_tokens = json_data['usage']['total_tokens']
                completion_tokens = json_data['usage']['completion_tokens']
                prompt_tokens = json_data['usage']['prompt_tokens']
                message = json_data["choices"][0]["message"]
                if process_id == 0:
                    print(f"[process({process_id})]total tokens: {json_data['usage']['total_tokens']}")
                if "function_call" in message.keys() and "." in message["function_call"]["name"]:
                    message["function_call"]["name"] = message["function_call"]["name"].split(".")[-1]
                return message, 0, total_tokens, completion_tokens, prompt_tokens
            except BaseException as e:
                print(f"[process({process_id})]Parsing Exception: {repr(e)}. Try again.")
                if json_data is not None:
                    print(f"[process({process_id})]OpenAI return: {json_data}")
            

        return {"role": "assistant", "content": str(json_data)}, -1, 0, 0, 0

if __name__ == "__main__":
    llm = ChatGPTFunction()
    prompt = '''下面这句英文可能有语病，能不能把语病都改掉？
If you think you get the result which can answer the task, call this function to give the final answer. Or, if you think you can't handle the task from this status, call this function to restart. Remember: you should ALWAYS call this function at the end of your try, and the final answer is the ONLY part that will be showed to user, so final answer should contain enough information.
没语病的形式：
'''
    messages = [
        {"role":"system","content":""},
        {"role":"user","content":prompt},
    ]
    llm.change_messages(messages)
    output,error_code,token_usage = llm.parse(functions=[],process_id=0)
    print(output)
