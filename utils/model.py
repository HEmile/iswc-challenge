import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def gpt3(prompts):
    """ functions to call GPT3 predictions """
    response = openai.Completion.create(
        model="text-davinci-002",
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
