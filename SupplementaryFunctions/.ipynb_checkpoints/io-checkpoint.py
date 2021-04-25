import networkx as nx
import pickle
import pathlib
import os


#####IO (INPUT/OUTPUT) FUNCTIONS#####

def PICKLE_LoadGraph(
    graphPath:str
) -> nx.Graph:
    '''
    Loads a graph from a .pkl checkpoint
    '''
    with open(graphPath, 'rb') as f:
        graph = pickle.load(f)
    print(f'Graph loaded from checkpoint: {graphPath}')
    return graph

def PICKLE_SaveGraph(
    graph:nx.Graph,
    filename:str,
) -> None:
    '''
    Saves a graph to a .pkl checkpoint
    '''
    graphPath = pathlib.Path(os.getcwd()).parent / (path.name + f'/Checkpoints/{filename}')
    with open(graphPath, 'wb') as f:
        pickle.dump(graph, f)
    print(f'Graph saved to checkpoint: {graphPath}')
    
def PICKLE_LoadSession(
    sessionPath:str
) -> dict:
    '''
    Loads a topic model from a .pkl checkpoint
    '''
    with open(sessionPath, 'rb') as f:
        session = pickle.load(f)
    print(f'Session loaded from checkpoint: {sessionPath}')
    return session

def PICKLE_SaveSession(
    session:dict,
    filename:str,
) -> None:
    '''
    Saves result of all intermediate variables to a .pkl checkpoint.
    '''
    path = pathlib.Path(os.getcwd())
    sessionPath = path.parent / (path.name + f'/Checkpoints/{filename}')
    with open(sessionPath, 'wb') as f:
        pickle.dump(session, f)
    print(f'Session saved to checkpoint: {sessionPath}')