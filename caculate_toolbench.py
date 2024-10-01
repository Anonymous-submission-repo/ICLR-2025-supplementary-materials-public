import os
import json

def calculate_usd(json_folder_path, model = "gpt4o"):
    win_true_count = 0
    win_false_count = 0
    total_count = 0
    win_true_tokens = 0
    win_false_tokens = 0
    
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    
    win_file = []
    size = {}
    for file_name in os.listdir(json_folder_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(json_folder_path, file_name)
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    total_count += 1
                    tokens = data.get('answer_generation', {}).get('total_tokens', 0)
                    
                    query_name = file_name.split("_")[0]
                    size[query_name] = data.get('tree', {}).get('size', 0)
                                        
                    total_tokens += tokens
                    input_tokens += data.get('answer_generation', {}).get('prompt_tokens', 0)
                    output_tokens += data.get('answer_generation', {}).get('completion_tokens', 0)
                    
                    if data.get('win'):
                        win_file.append(file_name.split("_")[0])
                        win_true_count += 1
                        win_true_tokens += tokens
                    else:
                        win_false_count += 1
                        win_false_tokens += tokens

    overall_avg_tokens = (total_tokens / total_count) if total_count > 0 else 0
    avg_input_tokens = (input_tokens / total_count) if input_tokens > 0 else 0
    avg_output_tokens = (output_tokens / total_count) if output_tokens > 0 else 0

    if model == "gpt4o":
        total_usd = avg_input_tokens /1000000 * 5 + avg_output_tokens /1000000 * 15 
    else:
        raise ValueError("Model not supported")

    return total_usd, size


path = "path_to_your_inference_result_folder"
total_usd, _ = calculate_usd(path)

print(total_usd)