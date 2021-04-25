from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from typing import Tuple
from collections import defaultdict

'LDA_bias_prefSemTypes'

#####TOPIC MODELS CALCULATION#####

def GENSIM_CalculateLDATopics(
    fullSession:dict,
) -> Tuple[dict, dict]:
    
    '''
    Function for calculating LDA topics given a session,
    such that:
        {'s:9628730:1:5': ['l:noun:phosphorylase',
                            'l:noun:insulin', 
                            'l:adj:sensitive',
                            'l:noun:synthase',...]}
    Thus,
        text_corpus = [[t1, t2, t3...], [t2, t3, t4...]]
        
    '''
    
    text_corpus = [
        fullSession['sentenceTokens'][key] for key in fullSession['sentenceTokens']
    ]
    
    def BiasLDAdocument(text_document, weightsDict, prefSemTypes:set = set()):
        
        result = []
        if prefSemTypes:
            for token in text_document:
                if token[0] == 'm' and fullSession['tokenSemTypes'][token].intersection(prefSemTypes):
                    for i in range(weightsDict['m']):
                        result.append(token)
                if token[0] in {'l', 'e', 'n'}:
                    for i in range(weightsDict[token[0]]):
                        result.append(token)
        else:
            for token in text_document:
                for i in range(weightsDict[token[0]]):
                        result.append(token)
        return result
    
    
    weightsDict = {
            'm': fullSession['params']['LDA_bias_mesh'],
            'l': fullSession['params']['LDA_bias_lemmas'],
            'e': fullSession['params']['LDA_bias_entities'],
            'n': fullSession['params']['LDA_bias_ngrams'],
        }
    
    
    prefSemTypes = set(fullSession['params']['LDA_bias_prefSemTypes'])
    print('Current preferred SemTypes: ', prefSemTypes)
    
    text_corpus_biased = [
        BiasLDAdocument(doc, weightsDict, prefSemTypes) for doc in text_corpus
    ]
    
    text_corpus_ids = [
        key for key in fullSession['sentenceTokens']
    ]
    
    print("Computing topics...")
    dictionary = Dictionary(text_corpus_biased)
    int_corpus = [dictionary.doc2bow(t) for t in text_corpus_biased]
    topic_model = LdaModel(
        corpus=int_corpus,
        id2word=dictionary,
        num_topics=fullSession['params']['LDA_nTopics'],
        random_state=42,
        iterations=50,
        passes=3,
        alpha=fullSession['params']['LDA_alpha'],
        eta=fullSession['params']['LDA_beta']
    )
    
    #Here we getting a dictionary like this:
    #{'topic_0': [
    #    ('l:noun:yeast', 0.023917323),
    #    ('l:noun:model', 0.018325549), ...]...}
    topicTerms = dict()
    for i in range(topic_model.num_topics):
        tempDict = dict()
        for top in topic_model.get_topic_terms(i):
            tempDict[dictionary[top[0]]] = top[1]
        topicTerms[f'topic_{i}'] = tempDict
        
    #Want to get most probable DOCUMENTS PER TOPIC
    print('Arranging documents per topic...')
    docsPerTopic = defaultdict(list)
    for i, int_doc in enumerate(int_corpus):
        for topicNum, prob in topic_model.get_document_topics(int_doc):
            docsPerTopic[f'topic_{topicNum}'].append(
                (text_corpus_ids[i], prob)
            )
    
    fullSession['topicTerms'] = topicTerms
    fullSession['docsPerTopic'] = docsPerTopic
    
    return topicTerms, docsPerTopic



def GetFormattedStringOfTopics(
    topicsTerms:dict, 
    key:str, 
    threshold:int = 10,
    termsDict:dict = dict(),
    max_len:int = 25,
) -> str:
    '''
    Returns a nicely formatted string to place on a shortest path graph.
    Number of top terms per topic to include is controlled by threshold parameter.
    '''
    try:
        t = topicsTerms[key]
    except:
        return ''
    
    nodeTopics = {
        k: v for k, v in sorted(
            topicsTerms[key].items(),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    nodeTopicsScores = [f'{key}:\n']
    for k in nodeTopics:
        if len(nodeTopicsScores) < threshold:
            if k[2:] in termsDict:
                term_str = termsDict[k[2:]]
                if len(term_str) > max_len:
                    term_str = f'{term_str[:max_len]}...'
                nodeTopicsScores.append(
                    ': '.join([f'{k[:3]}_{term_str}', str(nodeTopics[k])[:5]])
                )
            else:
                nodeTopicsScores.append(
                    ': '.join([k, str(nodeTopics[k])[:5]])
                )
    return('\n' + '\n#'.join(nodeTopicsScores) + '\n')