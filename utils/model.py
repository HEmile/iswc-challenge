import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def gpt3(prompts, model="text-davinci-002"):
    """ functions to call GPT3 predictions """
    response = openai.Completion.create(
        model=model,
        prompt=prompts,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=1
    )
    return [
        {
            'prompt': prompt,
            'text': response['text'],
            'tokens': response['logprobs']['tokens'],
            'logprob': response['logprobs']['token_logprobs']
        }
        for response, prompt in zip(response.choices, prompts)
    ]


def clean_up(probe_outputs):
    """ functions to clean up api output """
    probe_outputs = probe_outputs.strip()
    probe_outputs = probe_outputs[2:-2].split("', '")
    return probe_outputs


def convert_nan(probe_outputs):
    new_probe_outputs = []
    for item in probe_outputs:
        if item == 'None':
            new_probe_outputs.append('')
        else:
            new_probe_outputs.append(item)
    return new_probe_outputs
