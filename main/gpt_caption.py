# gpt/gpt_caption.py

import openai

def generate_caption(object_name):
    prompt = f"Write a short caption for an image containing a {object_name}:"

    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0.5,
    )

    caption = response.choices[0].text.strip()
    return caption
