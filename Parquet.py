# Use for training and evaluation
import pandas as pd
import torch
from PIL import Image
import numpy as np

# Load the parquet file
df = pd.read_parquet('results/answer_from_100_optimized.parquet')

# Convert to training format
def convert_to_training_data(df):
    training_examples = []
    
    for idx, row in df.iterrows():
        # Convert image array back to PIL Image
        image_array = np.array(row['image'], dtype=np.uint8)
        if image_array.shape[0] == 3:  # [C, H, W] format
            image_array = np.transpose(image_array, (1, 2, 0))  # [H, W, C]
        image = Image.fromarray(image_array)
        
        # Create training example
        example = {
            'image': image,
            'problem': row['problem'],
            'answer': row['answer'], 
            'solution': row['solution'],  # This is the high-quality MCTS solution
            'source': 'mcts_generated'
        }
        training_examples.append(example)
    
    return training_examples

training_data = convert_to_training_data(df)
print(f"Created {len(training_data)} training examples")

# Test VLM on the solved problems
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def test_vlm_on_mcts_data(model, processor, df, num_samples=5):
    results = []
    
    for idx in range(min(num_samples, len(df))):
        row = df.iloc[idx]
        
        # Prepare image
        image_array = np.array(row['image'], dtype=np.uint8)
        if image_array.shape[0] == 3:
            image_array = np.transpose(image_array, (1, 2, 0))
        image = Image.fromarray(image_array)
        
        # Create prompt
        prompt = f"Answer this math problem step by step: {row['problem']}"
        
        # Get VLM response
        message = [{
            "role": "user", 
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"}
            ]
        }]
        
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=image, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            vlm_response = processor.decode(generated_ids[0][inputs['input_ids'].shape[1]:], 
                                         skip_special_tokens=True)
        
        results.append({
            'problem': row['problem'],
            'mcts_solution': row['solution'],
            'vlm_response': vlm_response,
            'ground_truth': row['answer']
        })
    
    return results