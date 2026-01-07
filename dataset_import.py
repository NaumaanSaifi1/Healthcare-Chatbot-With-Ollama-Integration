from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

ds = load_dataset("heliosbrahma/mental_health_chatbot_dataset")
print (ds)
for split in ds:
    ds[split].to_json(f"mentalhealth_{split}.json")
