"""
Data preprocessing
"""
import argparse
import json
import os
from evaluation import ExecutionGraph,ExecutionNode
import random
random.seed(42)
parser = argparse.ArgumentParser()
parser.add_argument('--answer_dir',type=str, required= True, help='where the answers stored.')
parser.add_argument('--output', type=str, default=None, required=False, help='output path for the converted answer.')
parser.add_argument('--method',type=str, default=None, required=False, help='the name of the method.')

def generate_init_message_node(eg:ExecutionGraph,functions,query):
    init_node = ExecutionNode(role='system', message="You are AutoGPT, you can use many tools(functions) to do the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\nRemember: \n1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say \"I give up and restart\".\n2.All the thought is short, at most in 5 sentence.\n3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.\nLet's Begin!\nTask description: You should use functions to help handle the real time user querys. Remember to ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information to show to the user.\nSpecifically, you have access to the following functions: " + str(functions))
    eg.set_init_node(init_node)
    
    node = ExecutionNode(role='user', message=query)
    eg.add_node(node)
    eg[init_node,node] = None
    return node


def remove_selection(data):
    json_str = json.dumps(data)
    count = json_str.count('"name": "function_selection"')
    def helper(data):
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                if not (k == "message" and isinstance(v, dict) and v.get("name") == "function_selection"):
                    new_dict[k] = helper(v)
            return new_dict
        elif isinstance(data, list):
            new_list = []
            for item in data:
                if isinstance(item, dict) and "message" in item and isinstance(item["message"], dict) and item["message"].get("name") == "function_selection":
                    if "next" in item:
                        for next_item in item["next"]:
                            new_list.extend(helper([next_item]))
                else:
                    new_list.append(helper(item))
            return new_list
        else:
            return data
    cleaned_data = helper(data)
    return cleaned_data, count


def process_valid_data(method,answer_generation):
    conversation = answer_generation['train_messages'][-1]
    functions = answer_generation['function']
    query = answer_generation['query']
    eg = ExecutionGraph()
    last_node = generate_init_message_node(eg,functions,query)
    
    index = 2
    while index < len(conversation):
        message = conversation[index]
        role = message['role']
        if role == 'system' or role == 'user' or role == 'function':
            index = index + 1
            continue
        elif role == 'assistant':
            if 'function_call' in message :
                node = ExecutionNode(role='tool', message={
                    'name':message['function_call']['name'],
                    'arguments':message['function_call']['arguments'],
                    'response':conversation[index+1]['content'] if message['function_call']['name']!='Finish' else ''
                    })
                index = index + 1
            else:
                node = ExecutionNode(role='assistant',
                                        message=message['content'])
                
            
        else:
            raise NotImplementedError(f'Unkown role {role}')
        
        index = index + 1
        eg.add_node(node)
        eg[last_node,node] = None
        last_node = node
    
    eg = eg.reduce_graph_to_sequence()
    
    answer_details = eg.convert_to_dict()
    answer_details, count = remove_selection(answer_details)
    
    return {
        'query':query,
        'available_tools':functions,
        'answer':{
            'method':method,
            'total_steps': eg.node_count - count,
            'final_answer': answer_generation['final_answer'],
            'answer_details': answer_details
        }
    }
def process_invalid_data(method,data_dict):
    answer_generation = data_dict['answer_generation']
    functions = answer_generation['function']
    query = answer_generation['query']
    eg = ExecutionGraph()
    last_node = generate_init_message_node(eg,functions,query)
    if 'CoT' in method or 'cot' in method or "efficient" in method or "multi" in method:
        trail = random.choice(data_dict["trys"])

        
        index = 0
        while index < len(trail['chain']):
            message = trail['chain'][index]
            if message['node_type'] == 'Action':
                node = ExecutionNode(role='tool', message={
                    'name':message['description'],
                    'arguments':(trail['chain'][index+1]['description']),
                    'response':(trail['chain'][index+1]['observation'])})
            
                index = index + 1
            elif message['node_type'] == 'Thought':
                node = ExecutionNode(role='assistant',
                                        message=message['description'])
            else:
                raise NotImplementedError(f"Unknown node_type: {message['node_type']}")
            index = index + 1

            eg.add_node(node)
            eg[last_node,node] = None
            last_node = node
        eg = eg.reduce_graph_to_sequence()
   
    elif 'DFS' in method or 'dfs' in method:

        def DFS(root):
            if len(root['children']) == 0:
                node = ExecutionNode(role=root['node_type'],message=root)
                eg.add_node(node)
                return node
            else:
                child_nodes = [DFS(node) for node in root['children']]
                root['children'] = None
                root_node = ExecutionNode(role=root['node_type'],message=root)
                eg.add_node(root_node)
                for child_node in child_nodes:
                    eg.add_edge(root_node,child_node)
                return root_node
        for node in data_dict['tree']['tree']['children']:
            eg[last_node,DFS(node)] = None

        
        # purify the graph
        def purify_graph(node:ExecutionNode):
            if node.role == 'Action':
                adj_nodes = eg.get_adjacent_node(node)
                for adj_node in adj_nodes:
                    adj_node = eg[adj_node]
                    if adj_node.role == 'Action Input':
                        node.role = 'tool'
                        node.message = {
                            'name':node.message['description'],
                            'arguments':(adj_node.message['description']),
                            'response':(adj_node.message['observation'])
                            
                        }
                        # remove adj_node
                        adj_node = eg.pop_node(adj_node)
                        to_nodes = eg.edges.pop(adj_node.node_id,{})
                        eg.edges[node.node_id].update(to_nodes)
                        eg.edges[node.node_id].pop(adj_node.node_id)
                        node.out_degree += len(to_nodes)
                        break
            elif node.role == 'Thought':
                node.role = 'assistant'
                node.message = node.message['description']
            elif node.role == 'Action Input':
                print('Founding Extra Action Input Node')
                pass
            elif node.role =='system' or node.role=='user':
                pass
            else:
                raise Exception('Unknown role {}'.format(node.role))
            adj_nodes = eg.get_adjacent_node(node)
            for adj_node in adj_nodes:
                purify_graph(eg[adj_node])
            
        purify_graph(last_node)
        eg = eg.reduce_graph_to_sequence()
    else:
        raise NotImplementedError(f'Unknown method {method}')
    
    
    answer_details = eg.convert_to_dict()
    answer_details, count = remove_selection(answer_details)

    return {
        'query':query,
        'available_tools':functions,
        'answer':{
            'method':method,
            'total_steps': eg.node_count - count,
            'final_answer': answer_generation['final_answer'],
            'answer_details': answer_details
        }
    }
             

if __name__=='__main__':
    args = parser.parse_args()
    answer_dir = args.answer_dir
    output_dir = args.output
    method = args.method
    if "dfs" not in args.answer_dir and "DFS" not in args.answer_dir and method is None:
        method = answer_dir.split('/')[-2].split('_')[-1]
    elif ("dfs" in args.answer_dir or "DFS" in args.answer_dir) and method is None:
        method = answer_dir.split('/')[-2].split('_')[-3]
    if output_dir is None:
        answer_dir_list = answer_dir.split('/')
        replace_index = answer_dir_list.index('inference_result')
        answer_dir_list[replace_index] = 'converted_inference_result'
        output_dir = answer_dir_list[:-1]
        output_dir = '/'.join(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)        
        suffix = answer_dir_list[-1]+".json"
        output_dir = os.path.join(output_dir,suffix)
        args.output = output_dir
    output = args.output
    answer_dict = {}
    for filename in os.listdir(answer_dir):
        if filename.endswith('.json') and method in filename:
            qid = filename.split('_')[0]
            data_dict = json.load(open(os.path.join(answer_dir,filename)))
            if not data_dict['answer_generation']['valid_data']:
                answer_dict[qid] = process_invalid_data(method,data_dict)
            else:
                answer_dict[qid] = process_valid_data(method,data_dict['answer_generation'])
                
    json.dump(answer_dict,open(output,'w'))