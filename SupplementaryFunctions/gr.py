import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from typing import Tuple
import umap


#####CENTROIDS CALCULATION#####

def Gen_GetTopicsCentroids(
    #modelData['tokenCoordinates'],
    #model['topicsTerms'], 
    session:dict,
    #nTopicsContributingToCentroidCalculation:int = 15,
) -> None:
    '''
    Calculates centroid coordinates and radiuses for topics.
    '''
    
    source = session['source']
    target = session['target']
    
    print("Calculating topic centroid coordinates...")
    
    res = dict()
    resRads = dict()
    for i, topic in enumerate(session['topicTerms'].values()):
        vectors = []
        vectProbs = []
        
        #Sorting probabilities for tokens per topic
        topic = {
            k: v for k, v in sorted(
                topic.items(), 
                key = lambda item: item[1],
                reverse = True
            )
        }
        
        #Coordinates part
        for term in list(topic.keys())[:
            #:session['params']['KNN_nTopicsContributingToCentroidCalculation']
        ]:
            if type(session['tokenCoordinates'][term]) == np.ndarray:
                vectors.append(
                    session['tokenCoordinates'][term]*topic[term]
                )
                vectProbs.append(topic[term])
            else:
                True
        centroidCoords = sum(vectors)/sum(vectProbs)
        
        #Radiuses part
        sums = 0
        for v in vectors:
            sums += (np.linalg.norm(v - centroidCoords))**2

        centroidRadius = np.sqrt(sums/len(vectors))

        res[f'topic_{i}'] = centroidCoords
        resRads[f'topic_{i}'] = centroidRadius
    
    res[source] = session['tokenCoordinates'][source]
    res[target] = session['tokenCoordinates'][target]
    
    session['centroidsCoords'] = dict(
        {
            'Coordinates': res, 
            'Radiuses': resRads
        }
    )
    
    #return dict(
     #   {
     #       'Coordinates': res, 
     #       'Radiuses': resRads
     #   }
    #)
    
#####KNN GRAPH CONSTRUCTION#####

def SKLEARN_ConstructKNN_Graph(
    session:dict,
) -> nx.Graph:
    '''
    Constructs KNN graph, given an output from 
        Gen_GetTopicsCentroids['Coordinates'].
    Adjustable parameter - number of k nearest neighbors.
    Returns:
        networkx graph (UNdirected!), which we're going to display with Bokeh.
    '''
    
    print('Construct KNN network...')
    
    vectors = []
    for key in session['centroidsCoords']['Coordinates']:
        vectors.append(session['centroidsCoords']['Coordinates'][key])
    labels = list(session['centroidsCoords']['Coordinates'].keys())
    
    A = kneighbors_graph(vectors,
                         n_neighbors = session['params']['KNN_nNeighbors'],
                         mode = 'distance',
                         include_self = 'auto'
                        )
    
    A2 = pd.DataFrame(A.toarray(),
                      index = labels,
                      columns = labels
                     )
    Gt = nx.from_pandas_adjacency(A2, 
                                  create_using=nx.Graph)
    
    
    session['nxGraph'] = Gt
    
    return Gt


#Function for shortest path calculation
#Is used in callbacks
def NX_CalculateShortestPath(
    G:nx.Graph, 
    source:str, 
    target:str, 
    intermediate = None
) -> Tuple[list, set]:
    """
    Calculates shortest path between 2 nodes 
    (possibly through an intermediate one if specified)
    """
    
    #Check whether we have an intermediate node selected
    if intermediate:
        shortestPathNodesPart1 = nx.shortest_path(G, source, intermediate)
        shortestPathEdgesPart1 = set(
            zip(
                shortestPathNodesPart1, shortestPathNodesPart1[1:]
            )
        )
        shortestPathNodesPart2 = nx.shortest_path(G, intermediate, target)
        shortestPathEdgesPart2 = set(
            zip(
                shortestPathNodesPart2, shortestPathNodesPart2[1:]
            )
        )
        shortestPathEdges = shortestPathEdgesPart1.union(
            shortestPathEdgesPart2)
        
        shortestPathNodes = shortestPathNodesPart1
        for n in shortestPathNodesPart2:
            if n not in shortestPathNodes:
                shortestPathNodes.append(n)
    else:    
        shortestPathNodes = nx.shortest_path(G, source, target)
        shortestPathEdges = set(
            zip(
                shortestPathNodes, shortestPathNodes[1:]
            )
        )
    return shortestPathNodes, shortestPathEdges


def NX_ConstructHypothesisChainGraph(
    shortestPathEdges:set,
    scale:int = 5,
) -> Tuple[nx.Graph, dict]:
    '''
    Constructs a nx graph for each hypothesis for a nice visualisation
    and readibility.
    '''
    Gpath = nx.from_edgelist(shortestPathEdges)
    #Gpos = nx.spring_layout(Gpath, scale = 5)
    GposChain = dict()
    #We need to eliminate x-axis coordinate to make it a chain
    for i, key in enumerate(Gpath.nodes):
        GposChain[key] = [0, i*scale]
    return Gpath, GposChain

def SKLEARN_CalculateGraphLayout(session) -> None:
    '''
    Calculates layout for our hypothesis space network given coordinates of centroids.
    We perform dim reduction from 512 to 2 using a desired method (PCA for now).
    '''
    print('Calculating hypothesis space node arrangement...')
    
    nodenames = list(session['centroidsCoords']['Coordinates'].keys())
    
    vectors512 = [session['centroidsCoords']['Coordinates'][v] for v in nodenames]
    
    #pca = PCA(n_components=2)
    #pca.fit(vectors512)
    
    #vectors2 = pca.transform(vectors512)
    
    print('calculating UMAP...')
    reducer = umap.UMAP()
    vectors2 = reducer.fit_transform(vectors512)
    
    center_offset = np.mean(vectors2, axis=0)
    
    for i in range(len(vectors2)):
        vectors2[i] = vectors2[i] - center_offset
    
    layout = dict()
    for i, node in enumerate(nodenames):
        layout[node] = vectors2[i]
    
    session['nxLayout'] = layout
    return layout


def SKLEARN_PerformTokensCoordsPCA(session) -> None:
    '''
    Performs PCA transform of our tokens coords.
    Only once to achieve consistency.
    '''
    fullSetOfLabels = [token for token in session['tokenCoordinates']]
    fullSetOfCoordinates = [session['tokenCoordinates'][token] for token in session['tokenCoordinates']]
    
    #print('Performing PCA transformation for tokens...')
    #pca = PCA(n_components=2)
    
    #pca.fit(fullSetOfCoordinates)
    #embFullSetOfCoordinates = pca.transform(fullSetOfCoordinates)
    
    print('calculating UMAP for all entities...')
    reducer = umap.UMAP()
    embFullSetOfCoordinates = reducer.fit_transform(fullSetOfCoordinates)
    
    dictFullSetOfCoordinates = dict(zip(fullSetOfLabels, embFullSetOfCoordinates))
    
    session['tokenCoordinatesPCA'] = dictFullSetOfCoordinates
    
    
