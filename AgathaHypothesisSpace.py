            ##### IMPORTING BLOCK #####

from SupplementaryFunctions import (io,
                                    tm,
                                    gr,
                                    bo,
                                   )

#from sklearn.neighbors import kneighbors_graph

#from gensim.corpora import Dictionary
#from gensim.models.ldamodel import LdaModel

from bokeh.plotting import (figure, 
                            curdoc,
                            from_networkx)

from bokeh.layouts import (row, 
                           column, 
                           layout,
                           Spacer,
                          )

from bokeh.events import MouseWheel

from bokeh.models import (BoxSelectTool,
                          WheelZoomTool,
                          PanTool,
                          TapTool,
                          HoverTool,
                          Circle,
                          Rect,
                          MultiLine,  
                          Range1d, 
                          Button,
                          Slider,
                          ColumnDataSource,
                          Label,
                          Plot,
                          Text,
                          LabelSet,
                          CustomJS,
                          DataTable,
                          TableColumn,
                          Div,
                          FileInput,
                          GraphRenderer,
                          FuncTickFormatter,
                          TextInput,
                          Dropdown,
                          Select,
                          MultiChoice,
                         )
#from bokeh.core.enums import AutosizeMode
from bokeh.models.annotations import Title
from bokeh.models.widgets import HTMLTemplateFormatter
from bokeh.themes import built_in_themes

import pandas as pd
import networkx as nx
import numpy as np
from typing import Any, List, Optional, Tuple
import copy
from collections import defaultdict
from pathlib import Path
import pathlib
import os
import pickle
import json

from sklearn.decomposition import PCA
from sklearn.manifold import MDS



                ##### GLOBAL CONSTANTS AND VARIABLES #####

#modelDataPath = '/home/ityagin/Agatha_experiments/TopicModelsTuning/BokehServer/dataSource_alcohol_anemia.pkl'
#pathToGraph = '/home/ityagin/Agatha_experiments/TopicModelsTuning/BokehServer/network_alcohol_anemia.pkl'
#pathToSession = '/home/ityagin/Agatha_experiments/TopicModelsTuning/BokehServer/session_alcohol_anemia.pkl'
#pathToSession = '/home/ityagin/Agatha_experiments/TopicModelsTuning/BokehServer/fullSession_alcohol_anemia.pkl'
#pathToSession = '/home/ityagin/Agatha_experiments/TopicModelsTuning/BokehServer/fullSession_alcohol_anemia_correct_sentenceTokens.pkl'

pathToSession = '/home/ityagin/Agatha_experiments/TopicModelsTuning/BokehServer/session_SARS_deforestation.pkl'

#pathToTermsDict = 'mTermsDict.json'
#with open(pathToTermsDict) as f:
#    mDict = json.load(f)
    
#pathToTermsDict = 'mTermsDict_full.pkl'
#with open(pathToTermsDict, 'rb') as f:
#    mDict = pickle.load(f)

path = pathlib.Path(os.getcwd())
txtPath = path.parent / (path.name + '/Checkpoints/checkpoint_name.txt')    
with open(txtPath) as f:
    pathToSession = f.read().strip()
    print(f'File {pathToSession} loaded.')

locVars = dict()
locVars['savePath'] = ''

#This part will remain unchanged
#with open(modelDataPath, 'rb') as f:
#    modelData = pickle.load(f)

pathC = pathlib.Path(os.getcwd())
filePath = pathC.parent / (pathC.name + f'/Checkpoints/{pathToSession}')  
session = io.PICKLE_LoadSession(filePath)

mDict = session['mDict']

#gr.SKLEARN_PerformTokensCoordsPCA(session)

spColor = 'red'
defaultNodeColor = 'orange'
defaultEdgeColor = 'gray'

fontScaling = 2
fontConst = 0.8

rangeX = 5

hypRangeX = 2
hypRangeY = 30
hypLabelsFontSize = '10px'
hypSP_distanceBetweenNodes = 7
hypGlyphHeight = 5
hypGlyphWidth = 3

LDA_nTopics = 100
LDA_alpha = 0.001
LDA_beta = 0.001

KNN_nNeighbors = 4
KNN_nTopicsContributingToCentroidCalculation = 15

print_nTokensPerTopicSP = 5


                ##### GLOBAL STATE VARIABLES #####

        ### These variables contain the content of what we show.
        ### If we need to update something, 
        ### we should change what is stored in these variables. ###

    ## Hypothesis space part ##

#G = nx.florentine_families_graph()
#G = Gen_ConstructGraphFromScratch()
#G = PICKLE_LoadGraph(pathToGraph)
#PICKLE_SaveSession(session, pathToSession)

G = session['nxGraph']

#nxCoordinates = nx.spring_layout(G, k = 0.8, scale = 4)
nxCoordinates = gr.SKLEARN_CalculateGraphLayout(session)
shortestPathNodes = nx.shortest_path(G, session['source'], session['target'])
shortestPathEdges = set(zip(shortestPathNodes,shortestPathNodes[1:]))

edge_attrs = dict()
for start_node, end_node in G.edges:
    edge_color = defaultEdgeColor
    if ((start_node, end_node) in shortestPathEdges) or ((end_node, start_node) in shortestPathEdges):
        edge_color = spColor
        #print((start_node, end_node))
    edge_attrs[(start_node, end_node)] = edge_color

nx.set_edge_attributes(
    session['nxGraph'], 
    edge_attrs, 
    "edge_color"
)

## Graph part ##

plot = figure(
    #max_width=950, 
    #min_width = 800,
    width_policy = 'max',
    #sizing_mode = 'stretch_width',
)

bo.BOKEH_HypSpace_CustomizeFigureObj(figureObj = plot, 
                                     source = session['source'], 
                                     target = session['target'],
                                     rangeX = rangeX,
                                     session = session,
                                    )


def BOKEH_HypSpace_MakeGraphRenderer(
    session:dict,
    plot:figure,
) -> Tuple[GraphRenderer, ColumnDataSource]:
    '''
    Given a networkx graph to create a GraphRenderer object from,
    Creates this object and returns it + its labels to draw.
    '''
    
    #nxCoordinates = nx.spring_layout(session['nxGraph'], k = 0.8, scale = 4)
    nxCoordinates = gr.SKLEARN_CalculateGraphLayout(session)
    
    graph = from_networkx(session['nxGraph'], 
                          nxCoordinates)
    
    shortestPathNodes = nx.shortest_path(session['nxGraph'], session['source'], session['target'])
    shortestPathEdges = set(zip(shortestPathNodes,shortestPathNodes[1:]))

    edge_attrs = dict()
    for start_node, end_node in session['nxGraph'].edges:
        edge_color = defaultEdgeColor
        if ((start_node, end_node) in shortestPathEdges) or ((end_node, start_node) in shortestPathEdges):
            edge_color = spColor
            print((start_node, end_node))
        edge_attrs[(start_node, end_node)] = edge_color

    nx.set_edge_attributes(
        session['nxGraph'], 
        edge_attrs, 
        "edge_color"
    )

    coordinates = pd.DataFrame.from_dict(
        nxCoordinates,
        orient='index'
    ).rename(
        {0: 'x', 1: 'y'}, 
        axis = 'columns'
    )

    graph.node_renderer.data_source.data['nodesize'] = \
        [0.01*len(_) for _ in graph.node_renderer.data_source.data['index']]
    graph.node_renderer.data_source.data['nodecolors'] = \
        ['red' if nodeName in shortestPathNodes else 'orange' for nodeName in session['nxGraph'].nodes]

    coordinates['name'] = coordinates.index
    coordinates['name'] = coordinates['name'].apply(len)

    coordinates['nodenames'] = coordinates.index
    coordinates['nodenames'] = coordinates['nodenames'].apply(lambda x: x+'\n'+x)

    graph.node_renderer.glyph = Circle(
        #size = 'nodesize',
        radius = 'nodesize',
        fill_color = 'nodecolors',
        fill_alpha = 0.7,
    )
    graph.node_renderer.nonselection_glyph = graph.node_renderer.glyph

    graph.edge_renderer.glyph = MultiLine(
        line_color="edge_color", 
        line_dash="solid",
        line_alpha=0.8, 
        line_width=2
    )


    coordinatesCDS = ColumnDataSource(data=coordinates)
    coordinatesCDS.data['displayedNodenames'] = coordinatesCDS.data['index']
    coordinatesCDS_origin = copy.deepcopy(coordinatesCDS)

    labels = Text(x='x', y='y', text='displayedNodenames', 
                      #level='overlay',
                      text_align = 'center',
                      text_baseline ='middle',
                      text_font_style = 'bold',
                      text_line_height = 1,
                      text_font_size = str(fontConst/(plot.x_range.end - plot.x_range.start)) + "vw",
                      #source=coordinatesCDS,
                     )
    
    return graph, labels, coordinatesCDS

graph, labels, coordinatesCDS = BOKEH_HypSpace_MakeGraphRenderer(session, plot)

plot.renderers.append(graph)
plot.add_glyph(coordinatesCDS, labels)

    ## Hypothesis shortest path part ##

hypPlot = figure(
    title = f'Shortest path in detail',
    x_range=Range1d(-hypRangeX, hypRangeX), 
    y_range=Range1d(-1, hypRangeY),
    #sizing_mode='scale_height',
    sizing_mode='fixed',
    width = 300
)
hypPlot.xgrid.visible = False
hypPlot.ygrid.visible = False
hypPlot.xaxis.visible = False
hypPlot.yaxis.visible = False



            ##### CALLBACK FUNCTIONS #####

def Callback_BOKEH_ClickOnNode(attr, old, new) -> None:
    #print(f'Callback_BOKEH_ClickOnNode was activated. New: {new}')
    '''
    Callback function for Bokeh. 
    Uses a function NX_CalculateShortestPath for nodes/edges recolouring,
    given a node through which shortest path should go.
    '''
    if new == []:
        print('No intermediate node was selected!')
        shortestPathNodes, shortestPathEdges = gr.NX_CalculateShortestPath(
            G = session['nxGraph'],
            source = session['source'],
            target = session['target'],
            intermediate = None,
        )
    else:
        intermediateNode = graph.node_renderer.data_source.data['index'][new[0]]
        print(f'Calculating shortest path through {intermediateNode}')

        shortestPathNodes, shortestPathEdges = gr.NX_CalculateShortestPath(
            G = session['nxGraph'],
            source = session['source'],
            target = session['target'],
            intermediate = intermediateNode,
        )
    #print(f'''Shortest Path between {source} through {intermediateNode} to {target} is:\n {shortestPathNodes}
    #    \nEdges: {shortestPathEdges}''')
    
    tempEdges = graph.edge_renderer.data_source.data
    for i in range(len(tempEdges['edge_color'])):
        c = tempEdges['edge_color'][i]
        s = tempEdges['start'][i]
        e = tempEdges['end'][i]
        if ((s, e) in shortestPathEdges) or ((e, s) in shortestPathEdges):
            #print(f'''edge {(s, e)} was coloured {tempEdges['edge_color'][i]}''')
            tempEdges['edge_color'][i] = spColor
            #print(f'''Now edge is colored: {tempEdges['edge_color'][i]}''')
        else:
            tempEdges['edge_color'][i] = defaultEdgeColor
    graph.edge_renderer.data_source.data = dict(tempEdges)
    
    #Nodes FOR loop
    tempNodes = graph.node_renderer.data_source.data
    #tempCoordCDS = coordinatesCDS.data
    for i in range(len(tempNodes['nodecolors'])):
        if tempNodes['index'][i] in shortestPathNodes:
            tempNodes['nodecolors'][i] = spColor
            #tempCoordCDS['displayedNodenames'][i] = tempCoordCDS['nodenames'][i]
        else:
            #print(f'''index before renaming: {tempCoordCDS['index'][i]}''')
            tempNodes['nodecolors'][i] = defaultNodeColor
            #tempCoordCDS['displayedNodenames'][i] = coordinatesCDS_origin.data['index'][i]
    graph.node_renderer.data_source.data = dict(tempNodes)
    #coordinatesCDS.data = dict(tempCoordCDS)
    
    
    #Updating Shortest Path pane occurs here
    GpathNew, GposChainNew = gr.NX_ConstructHypothesisChainGraph(
        shortestPathEdges = list(zip(shortestPathNodes, shortestPathNodes[1:])),
        scale = hypSP_distanceBetweenNodes,
    )
    
    hypGraphNew = from_networkx(
        GpathNew, 
        GposChainNew, 
    )
    
    hypCoordinatesNew = pd.DataFrame.from_dict(
        GposChainNew,
        orient='index'
    ).rename(
        {0: 'x', 1: 'y'}, 
        axis = 'columns'
    )
    hypCoordinatesNew['TopicInfo'] = \
    hypCoordinatesNew.index.to_series().apply(
        lambda x: tm.GetFormattedStringOfTopics(
            session['topicTerms'], 
            x,
            print_nTokensPerTopicSP,
            termsDict=mDict,
        ) if 'topic' in x else x)
    
    hypCoordinatesCDSNew = ColumnDataSource(data=hypCoordinatesNew)
    hypCoordinatesCDS.data = dict(hypCoordinatesCDSNew.data)
    
    #global hypPlot1
    global hypGraph1
    
    #hypGraph1.node_renderer.data_source.data = dict(hypGraphNew.node_renderer.data_source.data)
    
    #hypGraph1.edge_renderer.data_source.data = dict(hypGraphNew.edge_renderer.data_source.data)
    
    hypPlot.renderers.remove(hypGraph1)
    #hypPlot.add_glyph(hypCoordinatesCDS, hypLabels)
    hypGraph1, _1, _2 = BOKEH_DrawHypothesisChainGraph(shortestPathNodes)
    
    #hypPlot.add_glyph(hypCoordinatesCDS, hypLabels)
    hypPlot.renderers.append(hypGraph1)
    hypGraph1.level = 'underlay'
    
    #Update Sentences per topic
    try:
        nName = graph.node_renderer.data_source.data['index'][new[0]]
    except:
        nName = ''
    if 'topic' in nName:
        print(f'Updading sentences per topic {nName}')
        
        dfSentPerTopicNew = pd.DataFrame(
            session['docsPerTopic'][nName], 
            columns = ['Sentence ID', 'Score']
        )
        #print('''session['docsPerTopic']:\n''', session['docsPerTopic'][nName][:5])
        #print(f"Is dfSentPerTopicNew empty? Len: {len(dfSentPerTopicNew)}")
       # print(dfSentPerTopicNew.head())
        
        #Error is somewhere there....
        dfSentPerTopic_show_new = dfSentPerTopicNew \
            .merge(dfSentTexts, how='left') \
            .merge(dfSentTokens, how = 'left', on = 'Sentence ID') \
            .sort_values(by='Score', ascending = False) 

        CDS_sentPerTopic_new = ColumnDataSource(dfSentPerTopic_show_new)

        #CDS_sentPerTopic.data = {}
        CDS_sentPerTopic.data = dict(CDS_sentPerTopic_new.data)
        dropdown_TopicPicker.value = nName
    
    global rend_circle_coordSpace
    
    plot_CoordinateSpace.renderers.remove(rend_circle_coordSpace)
    CDS_coordSpace_new, labels_coordSpace, rend_circle_coordSpace = Gen_FillCoordinateSpace(plot_CoordinateSpace, 
                                                                    session,
                                                                    shortestPathNodes)
    #plot_CoordinateSpace.renderers.append(rend_circle_coordSpace)
    CDS_coordSpace.data = dict(CDS_coordSpace_new.data)
    
    
    #circle_coordSpace.data_source.data = dict(circle_coordSpace_new.data_source.data)
    
        
            #print(f'IS IT EMPTY NOW??????\n{CDS_sentPerTopic.data}')
            #print(f'''Is df empty? {len(dfSentPerTopic_show_new)},\nsourceSize = {len(session['docsPerTopic'][nName])}''')
            #CDS_sentPerTopic.data = copy.deepcopy(dict(CDS_sentPerTopic_new.data))
    
    ## Recunstruct button callback, probably one of the key functions in our pipeline
    
def BOKEH_Callback_ReconstructButton() -> None:
    '''
    Reconstructs hypothesis space.
    Steps:
        (1) Clear plot and hypPlot (remove graph and hypGraph1)
        (2) With existing model['params'] rewrite the rest of model variables:
            topicTerms,
            docsPerTopic,
            centroidsCoords,
            nxGraph
        (3) Update CDS for:
            coordinatesCDS (plot var)
            CDS_sentPerTopic.data = {}
        (4) Attach graph and hyp
        (5) Add callback function (again)
            
            
    '''
    #CDS_sentPerTopic.data = {'Sentence ID': [' '], 'Score' : [' '], 'Sentence Text': ['Space reconstruction, please wait...']}
    
    print('Reconstructing hypothesis space with the following parameters:')

    print(session['params'])
    global graph
    global hypGraph1
    
    
    # (1)
    if graph in plot.renderers:
        plot.renderers.remove(graph)
        hypPlot.renderers.remove(hypGraph1)
    
    # (2)
    Gen_ConstructGraphFromScratch(session)
    gr.SKLEARN_CalculateGraphLayout(session)
    
    graph, labels, coordinatesCDS_new = BOKEH_HypSpace_MakeGraphRenderer(session, plot)
    
    shortestPathNodes = nx.shortest_path(session['nxGraph'], session['source'], session['target'])
    hypGraph1, hypCoordinatesCDS_new, hypLabels = BOKEH_DrawHypothesisChainGraph(shortestPathNodes)
    
    
    # (3)
    
    coordinatesCDS.data = dict(coordinatesCDS_new.data)
    hypCoordinatesCDS.data = dict(hypCoordinatesCDS_new.data)
    CDS_sentPerTopic.data = {}
    
    # (4)
    plot.renderers.append(graph)
    graph.level = 'underlay'
    
    hypPlot.renderers.append(hypGraph1)
    
    BOKEH_UpdateTopicsInfo(session, divTopics)
    dropdown_TopicPicker.options = [key for key in session['topicTerms'].keys()]
    
    # (5)
    graph.node_renderer.data_source.selected.on_change(
        "indices", 
        Callback_BOKEH_ClickOnNode,
    )
    
    global rend_circle_coordSpace
    
    plot_CoordinateSpace.renderers.remove(rend_circle_coordSpace)
    CDS_coordSpace_new, labels_coordSpace, rend_circle_coordSpace = Gen_FillCoordinateSpace(plot_CoordinateSpace, 
                                                                    session,
                                                                    shortestPathNodes)
    #plot_CoordinateSpace.renderers.append(rend_circle_coordSpace)
    CDS_coordSpace.data = dict(CDS_coordSpace_new.data)
    
    
            ##### GRAPH DEFENITION BLOCK #####

def Gen_ConstructGraphFromScratch(
    #That's where we put our results:
    session:dict
) -> None:
    '''
    Given: model data (from the previous part of the pipeline)
    Produces: nx.Graph which is shown in Hypothesis Space panel.
    Uses abovemention functions, just summarizes everything.
    '''
    topicTerms, docsPerTopic = tm.GENSIM_CalculateLDATopics(
        session,
    )
    #session['topicTerms'] = topicTerms
    #session['docsPerTopic'] = docsPerTopic
    
    centroidsCoords = gr.Gen_GetTopicsCentroids(
        session
    )
    
    #session['centroidsCoords'] = centroidsCoords
    
    nxGraph = gr.SKLEARN_ConstructKNN_Graph(
        #tc = centroidsCoords,
        #n_neighbors = session['params']['KNN_nNeighbors'],
        session
    )
    nx.set_edge_attributes(
        nxGraph, 
        "gray", 
        "edge_color"
    )
    session['nxGraph'] = nxGraph
    #Make things interesting:
    try:
        session['nxGraph'].remove_edge(session['source'], session['target'])
    except:
        print('Explicit edge between source and target does not exist, great!')
    



def BOKEH_DrawHypothesisChainGraph(shortestPathNodes):
    '''
    Fully takes care of ShortestPath graph
    '''
    Gpath, GposChain = gr.NX_ConstructHypothesisChainGraph(
        shortestPathEdges = list(zip(shortestPathNodes, shortestPathNodes[1:])),
        scale = hypSP_distanceBetweenNodes,
    )

    hypGraph = from_networkx(Gpath, 
                             GposChain, 
                            )

    #hypPlot.renderers.append(hypGraph)
    hypGraph.node_renderer.nonselection_glyph = hypGraph.node_renderer.glyph

    hypCoordinates = pd.DataFrame.from_dict(
        GposChain,
        orient='index'
    ).rename(
        {0: 'x', 1: 'y'}, 
        axis = 'columns'
    )
    hypCoordinates['TopicInfo'] = \
    hypCoordinates.index.to_series().apply(
        lambda x: tm.GetFormattedStringOfTopics(
            session['topicTerms'], 
            x,
            print_nTokensPerTopicSP,
            termsDict=mDict,
        ) if 'topic' in x else x)
    
    hypCoordinatesCDS = ColumnDataSource(data=hypCoordinates)

    hypLabels = Text(x='x', y='y', text='TopicInfo', 
                      text_align = 'center',
                      text_baseline ='middle',
                      text_font_style = 'bold',
                      text_line_height = 1,
                      text_font_size = hypLabelsFontSize,
                     )

    hypGraph.node_renderer.glyph = Rect(width = hypGlyphWidth,
                                        height = hypGlyphHeight,
                                        fill_color = 'red',
                                        fill_alpha = 0.8
                                       )

    #hypPlot.add_glyph(hypCoordinatesCDS, hypLabels)

    #hypWheel_zoom = WheelZoomTool(zoom_on_axis = False)

    hypPlot.tools = []
    hypPlot.add_tools(
        #hypWheel_zoom,
        HoverTool(tooltips=None), 
        #TapTool(),
        PanTool()
    )
    #hypPlot.toolbar.active_scroll = hypWheel_zoom
    hypPlot.toolbar_location = None
    
    return hypGraph, hypCoordinatesCDS, hypLabels

hypGraph1, hypCoordinatesCDS, hypLabels = BOKEH_DrawHypothesisChainGraph(
    nx.shortest_path(
        session['nxGraph'], session['source'], session['target'])
)

hypPlot.renderers.append(hypGraph1)
hypPlot.add_glyph(hypCoordinatesCDS, hypLabels)



            ##### TOPIC RELEVANT SENTENCES PLOT #####

dfSentTexts = pd.DataFrame.from_dict(
    session['sentenceTexts'].items(),
).rename(
    {0: 'Sentence ID', 1: 'Sentence Text'}, 
    axis = 'columns'
)

dfSentTokens = pd.DataFrame.from_dict(
    session['sentenceTokens'].items(),
).rename(
    {0: 'Sentence ID', 1: 'Sentence Tokens'}, 
    axis = 'columns'
)

#Creating empty dataframe, because from the very beginning
#nothing is selected
dfSentPerTopic = pd.DataFrame(
    #session['docsPerTopic']['topic_10'], 
    {},
    columns = ['Sentence ID', 'Score']
)
dfSentPerTopic_show = dfSentPerTopic \
    .merge(dfSentTexts, how='left') \
    .sort_values(by='Score', ascending = False)

CDS_sentPerTopic = ColumnDataSource(dfSentPerTopic_show)

data_table = DataTable(source=CDS_sentPerTopic, 
                       columns=[
                           TableColumn(field = 'Sentence ID',
                                       title = 'Sentence ID',
                                       width = 60,
                                      ), 
                           TableColumn(field = 'Score', 
                                       title = 'Score',
                                       width = 70,
                                      ), 
                           TableColumn(field = 'Sentence Text', 
                                       title = 'Sentence Text',
                                       width = 700,
                                       formatter = HTMLTemplateFormatter(
                                           template="""<span href="#" data-toggle="tooltip" title="<%= value %>"><%= value %></span>"""),
                                      ),
                           TableColumn(field = 'Sentence Tokens', 
                                       title = 'Sentence Tokens',
                                       width = 100,
                                       formatter = HTMLTemplateFormatter(
                                           template="""<span href="#" data-toggle="tooltip" title="<%= value %>"><%= value %></span>"""),
                                      )
                                ],
                       editable=True,
                       sizing_mode = 'stretch_width',
                       #AutosizeMode = "fit_columns",
                       height_policy='max',
                       #height=350
                      )



            ##### TOPIC TOKENS SECTION #####
    
def BOKEH_UpdateTopicsInfo(
    session:dict,
    divTopics:Div,
) -> None:
    divStr = '<p><b><u>List of topics:</u></b></p>'
    divStr += '<ul>'
    for key in list(session['topicTerms'].keys()):
        divStr += f'<li><b>{key}</b></li>'
        divStr += '<ul>'
        for token in session['topicTerms'][key]:
            if token[0] == 'm':
                try:
                    desc = mDict[token.split(':')[-1]]
                    tokenStr = token + ' - ' + desc
                except:
                    tokenStr = token
            else:
                tokenStr = token
            divStr += f'''<li>{tokenStr}: {str(session['topicTerms'][key][token])[:5]}</li>'''
        divStr += '</ul>'
        divStr += '<p></p>'
    divStr += '</ul>'
    
    divTopics.text = divStr


divTopics = Div(
    text = "",
    style={
        'overflow-y':'scroll',
        'height':'350px',
    }
)

BOKEH_UpdateTopicsInfo(session, divTopics)


            ##### BOKEH CALLBACKS BLOCK ##### 

callbackFontSize = CustomJS(
    args = dict(labelset=labels, pl = plot, fs = fontScaling, fc = fontConst), 
    code="""
    var currentZoom = (pl.x_range.end - pl.x_range.start);
    labelset.text_font_size = String(fs*fc/currentZoom) + "vw";
    if (currentZoom > 1)
    {
        labelset.text_line_height = fs*1/currentZoom;
    }
    labelset.change.emit();
    """)

graph.node_renderer.data_source.selected.on_change(
    "indices", 
    Callback_BOKEH_ClickOnNode,
)
plot.js_on_event(MouseWheel, callbackFontSize) 
     
    

            ##### BOKEH LAYOUT BLOCK ##### 
        
#buttons = [Button(label="Press Me") for _ in range(5)]
#menu_TopicPicker = [(key, key) for key in session['topicTerms'].keys()]
menu_TopicPicker = [key for key in session['topicTerms'].keys()]
#dropdown_TopicPicker = Dropdown(label="Pick a topic...", button_type="success", menu=menu_TopicPicker)
dropdown_TopicPicker = Select(
    title="Pick a topic...", 
    options=menu_TopicPicker,
    max_width=250,
    value = 'Select from this list'
)
dropdown_TopicPicker.on_change(
    'value', 
    lambda attr, old, new: Callback_BOKEH_ClickOnNode(attr, 
                                                      old, 
                                                      new = [int(new.split('_')[-1])]
                                                     )
)

    ## Coord space block ##

def Gen_FillCoordinateSpace(p, session, shortestPathNodes) -> None:
    
    embLabels = []
    embLabels_print = []
    #for topic in list(session['topicTerms'].keys())[:4]:
    for topic in shortestPathNodes:
        if 'topic' in topic:
            for term in session['topicTerms'][topic]:
                if term[:2] == 'm:' and term[2:] in mDict:
                    term_print = f'{term}_{mDict[term[2:]]}'
                else:
                    term_print = term
                embLabels.append(term)
                embLabels_print.append(term_print)
                
    
    #embLabels = []
    #for topic in shortestPathNodes:
    #for topic in list(session['topicTerms'].keys())[:]:    
        #if 'topic' in topic:
            #embLabels.append(topic)
    
    embLabels.append(session['source'])
    embLabels.append(session['target'])
    
    #embLabels_print.append('')
    #embLabels_print.append('')
    
    term1 = session['source']
    term2 = session['target']
    for term in [term1, term2]:
        if term[:2] == 'm:' and term[2:] in mDict:
            embLabels_print.append(f'{term}_{mDict[term[2:]]}')
        else:
            embLabels_print.append(term)
    #embLabels_print.append()
    
    #print('embLabels:', embLabels)
    
    #embFullCoords = []
    #for l in embLabels:
        #embFullCoords.append(session['tokenCoordinates'][l])
       # try:
         #   embFullCoords.append(session['centroidsCoords']['Coordinates'][l])
       # except:
         #   print(f'Could not find coordinates for: {l}')
            
    embCoords = np.array([session['tokenCoordinatesPCA'][key] for key in embLabels])
        
    #------#
    #pca.fit(embFullCoords)
    #embCoords = pca.transform(embFullCoords)
    
    #mds = MDS(n_components=2)
    #embCoords = mds.fit_transform(embFullCoords)
    
    def Colormap(key):
        #print('Colormap:', key)
        if (key == session['source']) or (key == session['target']):
            return 'black'
        if key[0] == 'n':
            return 'skyblue'
        if key[0] == 'm':
            return 'lightgray'
        if key[0] == 'e':
            return 'aquamarine'
        if key[0] == 'l':
            return 'plum'
        if 'topic' in key:
            return 'salmon'

    circle = p.circle(embCoords[:,0], 
                 embCoords[:,1],
                 color=[Colormap(_) for _ in embLabels], 
                 fill_alpha=0.9, 
                 size=10)
    #labels
    
    
    ldf = pd.DataFrame({
        'labels': embLabels_print,
        'x': embCoords[:,0],
        'y': embCoords[:,1]
    })
    
    source = ColumnDataSource(ldf)
    
    labels = LabelSet(x = 'x',
                      y = 'y',
                      y_offset=8,
                      text = 'labels',
                      source = source,
                      text_font_size="11px", 
                      text_color="#555555",
                     )
    
    return source, labels, circle

plot_CoordinateSpace = figure(title = 'Embeddings:',
                              #sizing_mode='scale_width',
                              width_policy='fit',
                              #sizing_mode = 'stretch_width',
                              #width_policy = 'max',
                              max_width = 400,
                             )
plot_CoordinateSpace.toolbar.active_scroll = plot_CoordinateSpace.select_one(WheelZoomTool)
plot_CoordinateSpace.toolbar_location = None

CDS_coordSpace, labels_coordSpace, rend_circle_coordSpace = Gen_FillCoordinateSpace(
    plot_CoordinateSpace, session, shortestPathNodes)
plot_CoordinateSpace.add_layout(labels_coordSpace)


#buttons = [plot_CoordinateSpace]
#buttons.append(dropdown_TopicPicker)

    ## Save/load block ##
    
def BOKEH_Callback_LoadSesionTextfieldHandler(attr, old, new) -> None:
    print("Updated label: " + new)
    
def BOKEH_Callback_SaveSesionTextfieldHandler(attr, old, new) -> None:
    print("Updated label: " + new)
    locVars['savePath'] = new
    print(f"Current savepath: {locVars['savePath']}")
    

def BOKEH_Callback_LoadSesionButtonHandler() -> None:
    print(f"Load Session button pressed. Performing Session loading from {pathToSession} ...")

def BOKEH_Callback_SaveSesionButtonHandler() -> None:
    print(f"Save Session button pressed. Performing Session saving to {locVars['savePath']} ...")
    try:
        io.PICKLE_SaveSession(session, locVars['savePath'])
    except Exception as e:
        print('''Couldn't save current session. Please check the output file correctness. ''')
        print(e)

# Elements:
input_showLoadSessionPath = TextInput(
    value="Enter input path...", title = 'Locate the checkpoint to restore session from',
    sizing_mode = 'stretch_width',)

input_showSaveSessionPath = TextInput(
    value="Enter checkpoint name...", title = 'Save current session to file:',
    sizing_mode = 'stretch_width',)
          
          
button_loadSession = Button(label='Load Session', max_width = 100)
button_saveSession = Button(label='Save Session', max_width = 100)

# Callbacks:
input_showLoadSessionPath.on_change("value", BOKEH_Callback_LoadSesionTextfieldHandler)
input_showSaveSessionPath.on_change("value", BOKEH_Callback_SaveSesionTextfieldHandler)

button_loadSession.on_click(BOKEH_Callback_LoadSesionButtonHandler)
button_saveSession.on_click(BOKEH_Callback_SaveSesionButtonHandler)


    ### Settings pane ###
    
## KNN settings ##
div_KNN = Div(
    text = "KNN&nbsp;settings: ",
    style = {'font-size': 'large'}
)

def BOKEH_UpdateParam(paramName: str, val:Any) -> None:
    '''
    Updates session['params']['paramName']
    '''
    session['params'][paramName] = val
    
    return None

slider_KNN_n_neighbors = Slider(start=2, end=15, value=session['params']['KNN_nNeighbors'], step=1, 
                                title="N neighbors")
slider_KNN_n_neighbors.on_change("value", 
                                 lambda attr, old, new: BOKEH_UpdateParam('KNN_nNeighbors', new))


slider_KNN_top_n_tokens = Slider(start=3, end=50, value=10, step=1, 
                                 title="Top N significant tokens in centroids coords")
slider_KNN_top_n_tokens.on_change("value", 
                                 lambda attr, old, new: BOKEH_UpdateParam('KNN_nTopicsContributingToCentroidCalculation', new))

## LDA settings ## 

div_LDA= Div(
    text = "LDA&nbsp;settings: ",
    style = {'font-size': 'large'},
    width_policy='max',
)

slider_LDA_n_topics = Slider(start=10, end=100, value=session['params']['LDA_nTopics'], step=1, 
                             title="Number of LDA topics")
slider_LDA_n_topics.on_change("value", 
                              lambda attr, old, new: BOKEH_UpdateParam('LDA_nTopics', new))


slider_LDA_alpha = Slider(start=-4, end=1, value=-2, step=1, 
                          title="LDA alpha parameter",
                          format=FuncTickFormatter(code="return (10**tick).toFixed(4)"))
slider_LDA_alpha.on_change("value", 
                           lambda attr, old, new: BOKEH_UpdateParam('LDA_alpha', 10**new))


slider_LDA_beta = Slider(start=-3, end=1, value=-2, step=1, 
                         title="LDA beta parameter",
                         format=FuncTickFormatter(code="return (10**tick).toFixed(4)"))
slider_LDA_beta.on_change("value", 
                           lambda attr, old, new: BOKEH_UpdateParam('LDA_beta', 10**new))


div_LDA_biases= Div(
    text = "LDA tokens relative importance: ",
    style = {'font-size': 'medium'}
)

slider_LDA_bias_mesh = Slider(start=1, end=50, value=session['params']['LDA_bias_mesh'], step=1, title="mesh/umls")
slider_LDA_bias_mesh.on_change("value", 
                              lambda attr, old, new: BOKEH_UpdateParam('LDA_bias_mesh', new))


slider_LDA_bias_lemmas = Slider(start=1, end=5, value=session['params']['LDA_bias_lemmas'], step=1, title="lemmas")
slider_LDA_bias_lemmas.on_change("value", 
                              lambda attr, old, new: BOKEH_UpdateParam('LDA_bias_lemmas', new))


slider_LDA_bias_entities = Slider(start=1, end=5, value=session['params']['LDA_bias_entities'], step=1, title="entities")
slider_LDA_bias_entities.on_change("value", 
                              lambda attr, old, new: BOKEH_UpdateParam('LDA_bias_entities', new))


slider_LDA_bias_ngrams = Slider(start=1, end=5, value=session['params']['LDA_bias_ngrams'], step=1, title="ngrams")
slider_LDA_bias_ngrams.on_change("value", 
                                 lambda attr, old, new: BOKEH_UpdateParam('LDA_bias_ngrams', new))


## SemTypes MultiChoice pane

semGroups = defaultdict(list)
with open('SemGroups_2018.txt') as f:
    for line in f:
        l = line.split('|')
        semGroups[l[1]].append(l[-1].strip())

LABELS = []
for k in semGroups:
    for semType_in_group in semGroups[k]:
        LABELS.append(f'{semType_in_group}')
LABELS.append('unknown')

multi_choice = MultiChoice(
    value=[], 
    options=LABELS, 
    title='Preferred Semantic Types: ',
    height=200,
    width = 300,
    height_policy='fixed',
    #css_classes=['Scrollable'],
)
multi_choice.js_on_change("value", CustomJS(code="""
    console.log('multi_choice: value=' + this.value, this.toString())
"""))

multi_choice.on_change("value", 
                       lambda attr, old, new: BOKEH_UpdateParam('LDA_bias_prefSemTypes', new))


## Reconstruct button ##

button_global_reconstruct = Button(label='Reconstruct Hypothesis Space')
button_global_reconstruct.on_click(BOKEH_Callback_ReconstructButton)
    

## Save/load interface

div_Save= Div(
    text = "<p></p>Save session: ",
    style = {'font-size': 'large'}
)
    ## Panes creation ##

#topLine = row(input_showSessionPath, button_loadSession, button_saveSession,
 #             height = 50,
 #             sizing_mode = 'stretch_width'
 #            )
    
settingsPane = column([div_KNN,
                       slider_KNN_n_neighbors, 
                       #slider_KNN_top_n_tokens,
                       div_LDA,
                       slider_LDA_n_topics,
                       slider_LDA_alpha,
                       slider_LDA_beta,
                       div_LDA_biases,
                       slider_LDA_bias_mesh,
                       slider_LDA_bias_lemmas,
                       slider_LDA_bias_entities,
                       slider_LDA_bias_ngrams,
                       button_global_reconstruct,
                       
                       div_Save,
                       #input_showLoadSessionPath,
                       #button_loadSession,
                       
                       input_showSaveSessionPath,
                       button_saveSession,
                       
                       multi_choice,
                      ],
                      #width = 200,
                     )

#dataTableLayout = column(data_table, topLine)

#rightPane = column(plot_CoordinateSpace, max_width=200, sizing_mode = 'stretch_width',)

leftPane = column(
    row(
        hypPlot, plot, plot_CoordinateSpace, #width_policy = 'max',
        sizing_mode = 'stretch_width',
    ), 
    row(
        row(data_table, sizing_mode = 'stretch_width'), 
        column(dropdown_TopicPicker, divTopics)
    ), 
    sizing_mode = 'stretch_width',
    height_policy = 'min'
)


    ## Layout creation ##
    
generalLayout = layout(row(leftPane, settingsPane, Spacer(width=10)),
                       sizing_mode = 'stretch_width',
                       #max_width = 800,
                       #width_policy='fit',
                      )

#####BOKEH SERVER BLOCK##### 

curdoc().add_root(generalLayout)
curdoc().title = "Agatha Semantic Visualiser"