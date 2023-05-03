import re

URL_TOKEN = '$URL'


def replace_url(sentence: str) -> str:
    search = re.findall('http://\S+|https://\S+', sentence)
    for i in search:
        sentence = sentence.replace(i, URL_TOKEN)

    return sentence


def replace_temporals(text: str) -> str:
    zwz_token = "Zwischenzeitlich"
    text = text.replace("In der Zwischenzeit", zwz_token)
    text = text.replace("in der Zwischenzeit", zwz_token)
    text = text.replace("in der zwischenzeit", zwz_token)

    return text
