from os.path import dirname, join

import pandas as pd
import re
from collections import defaultdict, Counter


from bokeh.models import CheckboxGroup, CustomJS, MultiChoice, TextInput
from bokeh.models.annotations import Title
from bokeh.models.widgets import HTMLTemplateFormatter
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, CustomJS, DataTable,
                          NumberFormatter, RangeSlider, TableColumn,)

from SupplementaryFunctions import (
    dt,
    io
)

import pathlib
import os
import pickle
import json

    ## Opening current ckpt ##

pathToSession = io.BOKEH_GetVckptFromURL(curdoc().session_context)

if not pathToSession:
    path = pathlib.Path(os.getcwd())
    txtPath = path.parent / (path.name + '/Checkpoints/checkpoint_name.txt')    
    with open(txtPath) as f:
        pathToSession = f.read().strip()
        print(f'File {pathToSession} loaded.')


pathC = pathlib.Path(os.getcwd())
filePath = pathC.parent / (pathC.name + f'/Checkpoints/{pathToSession}')  
session = io.PICKLE_LoadSession(filePath)

#with open('CUI_to_semtype_dict.pkl', 'rb') as f:
#    CUI_to_semtype = pickle.load(f)

    ## Working with DataFrames to show ##

#df = pd.read_csv(join(dirname(__file__), 'salary_data.csv'))

sent_tokens_counts = Counter([term for sent in session['sentenceTokens'].values() for term in sent])

source_tokens = ColumnDataSource(data=dict())
df_tokens = session['tokenSemTypes_df']
df_tokens['SemTypes'] = df_tokens['SemTypes'].apply(
    lambda x: x.union({'unknown'}) if len(x) == 0 else x,
)

df_tokens['SemTypes'] = df_tokens['SemTypes'].apply(
    #lambda x: ', '.join(x),
    lambda x: list(x),
)
df_tokens['Counter'] = df_tokens['Token'].apply(
    lambda x: sent_tokens_counts[x] if x in sent_tokens_counts else 0
)
df_tokens = df_tokens.sort_values(by='Counter', ascending = False) 

source = ColumnDataSource(data=dict())

df_sentences = pd.DataFrame.from_dict(
    session['sentenceTexts'].items(),
).rename(
    {0: 'Sentence ID', 1: 'Sentence Text'}, 
    axis = 'columns'
)

df_sent_tokens = pd.DataFrame.from_dict(
    session['sentenceTokens'].items(),
).rename(
    {0: 'Sentence ID', 1: 'Sentence Tokens'}, 
    axis = 'columns'
)

df_sentences_show = df_sentences \
    .merge(df_sent_tokens, how='left', on = 'Sentence ID')
df_sentences_show['Sentence Tokens'] = df_sentences_show['Sentence Tokens'] \
    .apply(lambda x: '; '.join(x) if type(x) == list else '')


    ## Callback function ##

def update():
    current = df_sentences_show
    #source.data = {
    #    'name'             : current.name,
    #    'salary'           : current.salary,
    #    'years_experience' : current.years_experience,
    #}
    source.data = current
    
def Search_handler() -> None:
    print('Button `button_search` pressed.')
    print(text_input.value)
    
    ## updating tokens list DataTable ##
    if multi_choice.value:
        current_multi_choice = set(multi_choice.value)
    else:
        current_multi_choice = set(LABELS)
    print('Current Semantic types:', current_multi_choice)
    
    temp_sent_tokens_df = \
        df_tokens[
            (df_tokens['Token_ext'].str.contains(
                re.escape(text_input.value.lower()))
            ) & \
            #(df_tokens['SemTypes'].str.contains('|'.join(multi_choice.value)))
            (df_tokens.SemTypes.map(
                lambda x: True if current_multi_choice.intersection(set(x)) else False)
            )
    ]
    
    current_tokens = '|'.join(set([re.escape(_) for _ in temp_sent_tokens_df['Token']]))
    
    source_tokens.data = temp_sent_tokens_df[:1000]
    
    
    ## updating sentence list DataTable ##
    
    temp_sent_df = \
        df_sentences_show[df_sentences_show['Sentence Tokens'].str.contains(current_tokens)]
    source_sentences.data = temp_sent_df
    
    return None
    
    ## Settings checkboxes ##
    
slider = RangeSlider(title="Max Salary", start=10000, end=110000, value=(10000, 50000), step=1000, format="0,0")
slider.on_change('value', lambda attr, old, new: update())


    ## Data sources ##

source_sentences = ColumnDataSource(data=df_sentences_show)

data_table = DataTable(
    source=source_sentences, 
    columns=[
        TableColumn(field = 'Sentence ID',
                   title = 'Sentence ID',
                   width = 60,
                   #formatter = HTMLTemplateFormatter(
                   #    template="""<a href="<%= value %>“target="https://pubmed.ncbi.nlm.nih.gov/<%= value %>"><%= value %>"""),
                  ), 
        TableColumn(field = 'Sentence Text', 
                   title = 'Sentence Text',
                   width = 700,
                   #formatter = HTMLTemplateFormatter(
                   #    template='''<a href="https://pubmed.ncbi.nlm.nih.gov/<%= PMID %>" target="_blank"><%= value %></a>'''),
                  ),
        TableColumn(field = 'Sentence Tokens', 
                   title = 'Sentence Tokens',
                   width = 100,
                   formatter = HTMLTemplateFormatter(
                       template="""<span href="#" data-toggle="tooltip" title="<%= value %>"><%= value %></span>"""),
                  )
            ],
    height_policy='max',
    width_policy='max',
    height=800,
    #width=1200,
    editable=True,
)

source_tokens = ColumnDataSource(data=df_tokens[:1000])

tokens_table = DataTable(
    source=source_tokens, 
    columns=[
        TableColumn(field = 'Token',
                   title = 'Token',
                   width = 70,
                  ), 
        TableColumn(field = 'Token_ext', 
                   title = 'Token_ext',
                   width = 300,
                   formatter = HTMLTemplateFormatter(
                       template="""<span href="#" data-toggle="tooltip" title="<%= value %>"><%= value %></span>"""),
                  ),
        TableColumn(field = 'SemTypes', 
                   title = 'Semantic Types',
                   width = 150,
                   formatter = HTMLTemplateFormatter(
                       template="""<span href="#" data-toggle="tooltip" title="<%= value %>"><%= value %></span>"""),
                  ),
        TableColumn(field = 'Counter', 
                   title = 'Counter',
                   width = 80,
                   ),
            ],
    height_policy='max',
    height=800,
    #width=600,
    editable=True,
)

    ## Download button ##

button_download = Button(label="Download", button_type="success")
button_download.js_on_click(
    CustomJS(args=dict(source=source_sentences),
             code=open(join(dirname(__file__), "download.js")).read()))

    ## Semantic Groups and Types ##

semGroups = defaultdict(list)
with open('SemGroups_2018.txt') as f:
    for line in f:
        l = line.split('|')
        semGroups[l[1]].append(l[-1].strip())

    
    ## `controls` column ##

LABELS = []
for k in semGroups:
    for semType_in_group in semGroups[k]:
        LABELS.append(f'{semType_in_group}')
LABELS.append('unknown')

#checkbox_group = CheckboxGroup(labels=LABELS, active=[0, 1])
#checkbox_group.js_on_click(CustomJS(code="""
#    console.log('checkbox_group: active=' + this.active, this.toString())
#"""))

multi_choice = MultiChoice(
    value=[], 
    options=LABELS, 
    title='Allowed Semantic Types: ',
    height=200,
    width = 270,
    height_policy='fixed',
    #css_classes=['Scrollable'],
)
multi_choice.js_on_change("value", CustomJS(code="""
    console.log('multi_choice: value=' + this.value, this.toString())
"""))

text_input = TextInput(value="", title="Term Search Box:")
text_input.js_on_change("value", CustomJS(code="""
    console.log('text_input: value=' + this.value, this.toString())
"""))

button_search = Button(label="Search", button_type="success")
button_search.on_click(Search_handler)

controls = column(
    text_input,
    #checkbox_group,
    #slider, 
    button_search,
    button_download,
    multi_choice,
    #height = 800,
    width = 300,
    #sizing_mode = 'fit',
    #css_classes=['scrollable'], sizing_mode = 'fixed',
)

curdoc().add_root(
    row(
        controls, tokens_table, data_table, 
        height_policy='max',
    )
)
curdoc().title = "Agatha Corpus Explorer"

update()
