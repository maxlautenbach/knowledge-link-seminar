from transformers import AutoModelForCausalLM, AutoTokenizer
from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

prompts = [
    "Steve Jobs was the founder of",
    "Olaf Scholz is the cancellor of",
    "Mannheim is a City in the country of"
]

generation_prompts = prompts

ground_truth = [
    'Apple',
    'Germany',
    'Germany'
                ]
target_new = [
    'Microsoft',
    'Austria',
    'Bavaria'
              ]
subject = [
    'Steve Jobs',
    'Olaf Scholz',
    'Mannheim'
            ]

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to('cuda')
batch = tokenizer(generation_prompts, return_tensors='pt', padding=True)

pre_edit_outputs = model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
    max_new_tokens=30
)


hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama3.2-1b.yaml')
editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
    keep_original_weight=False
)

print('*'*100)

post_edit_outputs = edited_model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
    max_new_tokens=30
)
print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])