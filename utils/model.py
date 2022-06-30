import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def gpt3(prompt):
    """ functions to call GPT3 predictions """
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=1
    )
    return [
        {
            'text': i['text'],
            'tokens': i['logprobs']['tokens'],
            'logprob': i['logprobs']['token_logprobs']
        }
        for i in response.choices
    ]

def clean_up(probe_outputs):
    """ functions to clean up api output """
    probe_outputs = probe_outputs.strip()
    probe_outputs = probe_outputs[2:-2].split("', '")
    return probe_outputs
