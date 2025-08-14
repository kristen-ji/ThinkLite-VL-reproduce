from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig
from PIL import Image
import requests
from io import BytesIO

# Download and load the image
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

instruct_prompt = r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "russwang/ThinkLite-VL-7B", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("russwang/ThinkLite-VL-7B")

greedy_generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=2048
)

# Load the image from URL
image = load_image_from_url("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")

messages = [
    
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": "Describe this image." + instruct_prompt},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = processor(
    text=text,
    images=image,
    padding=True,
    return_tensors="pt",
).to("cuda")

output = model.generate(
    **inputs,
    generation_config=greedy_generation_config,
    tokenizer=processor.tokenizer
)
output_text = processor.decode(
    output[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(output_text) 