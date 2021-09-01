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
                          CustomJS,
                          DataTable,
                          TableColumn,
                          Div,
                          FileInput,
                          GraphRenderer,
                          FuncTickFormatter,
                          TextInput,
                         )

from bokeh.models.annotations import Title

def BOKEH_HypSpace_CustomizeFigureObj(
    figureObj:figure,
    source:str,
    target:str,
    rangeX:float,
    session:dict,
) -> None:
    '''
    Applies all necessary stuff to bokeh figure object.
    We use it in Hypothesis space visualisation.
    '''
    hypSpaceTitle = Title()
    
    try:
        source = session['mDict'][source[2:]]
        target = session['mDict'][target[2:]]
    except:
        pass
    hypSpaceTitle.text = \
        f'''Hypothesis space of {source} --> {target} (Agatha score: {str(session['agathaScore'])[:5]})'''
    
    figureObj.title = hypSpaceTitle
    figureObj.x_range = Range1d(-rangeX, rangeX)
    figureObj.y_range = Range1d(-rangeX, rangeX)
    figureObj.sizing_mode = 'stretch_both'
    
    
    #figureObj.x_range.min_interval = 0.5
    #figureObj.x_range.max_interval = 10 
    
    #plot.xgrid.visible = False
    #plot.ygrid.visible = False
    #plot.xaxis.visible = False
    #plot.yaxis.visible = False

    #wheel_zoom = WheelZoomTool(zoom_on_axis = False)
    wheel_zoom = WheelZoomTool()

    figureObj.tools = []
    figureObj.add_tools(
        wheel_zoom,
        HoverTool(tooltips=None), 
        TapTool(),
        PanTool()
    )
    figureObj.toolbar.active_scroll = wheel_zoom


def BOKEH_Save_session_part_to_text(
    session:dict,
    shortestPathNodes: list,
) -> str:
    """
    Saves session variables to a text file.
    """
    str_results = ''
    
    str_results += 'AGATHA SEMANTIC VISUALIZER TEXT OUTPUT\n\n'
    
    str_results += f"source term: {session['source']}\n"
    str_results += f"target term: {session['target']}\n\n"
    
    str_results += f"Session parameters:\n\n"
    
    str_results += '\n'.join([str(_) for _ in session['params'].items() if 'Contributing' not in str(_)])
    
    str_results += '\n\n'
    
    str_results += "Selected shortest path:\n\n\t"
    str_results += " -> ".join(shortestPathNodes)
    
    str_results += '\n\nList of topics:\n\n'
    
    divStr = ''
    for key in list(session['topicTerms'].keys()):
        divStr += f"{key}\n"
        for token in session['topicTerms'][key]:
            if token[0] == 'm':
                try:
                    desc = session['mDict'][token.split(':')[-1]]
                    tokenStr = token + ' - ' + desc
                except:
                    tokenStr = token
            else:
                tokenStr = token
            divStr += f'''\t{tokenStr}: {str(session['topicTerms'][key][token])[:5]}\n'''
    
    str_results += divStr
    
    str_results += '\n\nTopical network (edgelist format):\n\n'
    str_results += '\n'.join([str(edge) for edge in list(session['nxGraph'].edges())])
    
    return str_results 
    
    
    
    

