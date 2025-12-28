import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from transformers import Pipeline, pipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM
    return AutoModelForCausalLM, AutoTokenizer, pipeline


@app.cell
def _(pipeline):
    pipe = pipeline("text-generation", model="Qwen/Qwen3-4B")
    return


@app.cell
def _(pipeline):
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    output = pipeline(messages)
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer):
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    inputs = tokenizer.apply_chat_template(
    	messages,
    	add_generation_prompt=True,
    	tokenize=True,
    	return_dict=True,
    	return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
    return


if __name__ == "__main__":
    app.run()
