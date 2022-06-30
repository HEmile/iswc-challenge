import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def gpt3(prompt):
    """ functions to call GPT3 predictions """
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=20,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=1
    )
    return response.choices[0]['text'], response.choices[0]['logprobs']['tokens'], response.choices[0]['logprobs'][
        'token_logprobs']

