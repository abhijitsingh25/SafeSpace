from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("alibayram/medgemma-4b")
lora = PeftModel.from_pretrained(base, "./exams_lora")

lora = lora.merge_and_unload()
lora.save_pretrained("./medgemma-mentalhealth-exams-merged")

print("Merged model saved to ./medgemma-mentalhealth-exams-merged")
