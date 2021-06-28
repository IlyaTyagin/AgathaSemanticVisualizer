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


