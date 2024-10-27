import utils

rome_test = utils.RomeTest("./hparams/ROME/gpt2-xl.yaml")
prompt = "Mannheim is a City in the country of"
max_effect_layer, max_effect_tensor, hidden_flow = rome_test.get_hidden_flow(prompt)
print(max_effect_layer)
print(max_effect_tensor)
rome_test.plot_causal_tracing_effect(prompt, kind="mlp", hidden_flow=hidden_flow)