import pandas as pd
import spacy


def make_bulletpoints(file, petri_path=None):
    nlp = spacy.load('de_dep_news_trf')
    # function to save constraints to list of bulletpoints.
    constraints = pd.read_json(f'output/{file}_constraints.jsonl', lines=True)

    og_text = pd.read_csv(f'output/{file}_i2sent.csv').sent.values
    with open(f'output/markdowns/{file}.md', 'w+') as f:
        f.write(f'# {file} \n\n')
        f.write(f'## Original Text:\n\n')
        for sent in og_text:
            if sent in constraints.text.values:
                f.write(f'<span style="color:green">{sent}</span> \n')
            else:
                f.write(f'<span style="color:red">{sent}</span> \n')

        f.write('\n')
        f.write('## Summary: \n\n')

        for sent in og_text:
            if sent in constraints.text.values:
                doc = nlp(sent)
                tokens = [token.text for token in doc]
                line = '- ' + " ".join(tokens).replace('.', '').strip() + '\n'
                f.write(line)

        f.write('\n\n')
        f.write('## Petri-Net: \n\n')
        f.write(f'<img title="a title" alt="Alt text" src="{petri_path}">')
