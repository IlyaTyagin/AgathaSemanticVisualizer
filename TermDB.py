import pandas as pd
import sqlite3

import re
from collections import defaultdict, Counter

from bokeh.models import CheckboxGroup, CustomJS, MultiChoice, TextInput
from bokeh.models.annotations import Title
from bokeh.models.widgets import HTMLTemplateFormatter
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, CustomJS, DataTable,
                          NumberFormatter, RangeSlider, TableColumn, Div)

import pathlib
import os
import pickle
import json



    ## Opening sqlite DB ##
    
db_connection = sqlite3.connect('agatha_c_UMLS_terms.sqlite')
db_cursor = db_connection.cursor()

sql_search_query = \
    """
    SELECT 
        *
    FROM 
        agatha_c_CUIs
    WHERE
        UPPER(STR) LIKE UPPER('%{}%')
    LIMIT 1000;
    """

# Dummy search term for initial empty table
search_term = '----------------------------'

search_results_df = pd.DataFrame(columns=['STR', 'CID', 'Semantic Types'])
#search_results_df = pd.DataFrame({'STR': [' '*400], 'CID': [' '*10], 'Semantic Types': [' '*20]})

search_results_cdf = ColumnDataSource(data=search_results_df)

data_table = DataTable(
    source=search_results_cdf, 
    columns=[
        TableColumn(field = 'STR',
                   title = 'STR',
                   width = 1000,
                   formatter = HTMLTemplateFormatter(
                       template="""<span href="#" data-toggle="tooltip" title="<%= value %>"><%= value %></span>"""),
                  ),
        TableColumn(field = 'CUI', 
                   title = 'CUI',
                   width = 100,
                   formatter = HTMLTemplateFormatter(
                       template="""<span href="#" data-toggle="tooltip" title="<%= value %>"><%= value %></span>"""),
                  ),
        TableColumn(field = 'Semantic Type', 
                   title = 'Semantic Type',
                   width = 200,
                   formatter = HTMLTemplateFormatter(
                       template="""<span href="#" data-toggle="tooltip" title="<%= value %>"><%= value %></span>"""),
                  )
            ],
    height_policy='max',
    width_policy='fit',
    #height=800,
    width=1200,
    align='center',
    autosize_mode='none',
    #sizing_mode='scale_both',
)

def button_upd() -> None:
    button_search.label = 'Please wait...'
    curdoc().add_next_tick_callback(Search_handler)

def Search_handler() -> None:
    """
    Handles search button press.
    """
    
    print('Button `button_search` pressed.')
    
    search_term = text_input.value
    
    print(search_term)
    
    search_results_new_df = pd.read_sql_query(
        sql_search_query.format(search_term), 
        con=db_connection
    ).drop(columns=['index'])
    
    search_results_cdf.data = search_results_new_df
    
    button_search.label = 'Search'
    
    return None


text_input = TextInput(value="", width_policy='max')
text_input.js_on_change("value", CustomJS(code="""
    console.log('text_input: value=' + this.value, this.toString())
"""))

button_search = Button(label="Search", button_type="success")
button_search.on_click(button_upd)


text_div = \
    """<p style=text-align:center>AGATHA Term Database</p>"""

div_title = Div(
    text=text_div,
    style={'font-size': '400%'}
)

curdoc().add_root(
    column(
        div_title,
        row(text_input, button_search),
        data_table,
        height_policy='max',
        width_policy='max',
        align='center'
    )
)
curdoc().title = "Agatha Term Search"