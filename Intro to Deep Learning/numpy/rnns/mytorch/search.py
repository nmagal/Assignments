import numpy as np


def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """

    # Follow the pseudocode from lecture to complete greedy search :-) 
    number_of_symbols, sequence_length, batch_size = y_probs.shape
    y_probs = y_probs.squeeze()
    
    forward_prob = np.prod(np.max(y_probs, axis=0))
    
    #Decoding the max forward probabilities into characters
    argmax = np.argmax(y_probs, axis=0)
    int_mapping = (np.arange(number_of_symbols))+1
    
    my_mapping= {}
    my_mapping[0] = '-'
    for key, value in zip(int_mapping, SymbolSets):
        my_mapping[key] = value
    
    decoded_string = ''
    for max_prob in argmax:
        decoded_string = decoded_string + my_mapping[max_prob]
        
    compressed_string =''
    for char in range(len(decoded_string)):
        if decoded_string[char] =='-' or (decoded_string[char] == decoded_string[char-1] and char > 0):
            pass
        else:
            compressed_string = compressed_string + decoded_string[char]

    return (compressed_string, forward_prob)

##############################################################################

PathScore = dict()
BlankPathScore = dict()

def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
    # Follow the pseudocode from lecture to complete beam search :-)
    number_of_symbols, sequence_length, batch_size = y_probs.shape
    y_probs = y_probs.squeeze()
    
    #Below gives us our initial paths, and then a dictionary with paths and scores of paths 
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets, y_probs[:, 0])

    for time_step in range(1,sequence_length):
        
        global BlankPathScore
        global PathScore
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol,NewBlankPathScore, NewPathScore, BeamWidth)
        
        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:,time_step])
        
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSets, y_probs[:,time_step])

    
    print("Now it is time to merge our paths")

    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)
    

    
    return (max(FinalPathScore, key=FinalPathScore.get), FinalPathScore)
    



def InitializePaths(SymbolSet, y):
    
    #These are our dictionaries holding our paths and scores. One ending in blank, other ending in symbol
    InitialBlankPathScore = dict() 
    InitialPathScore = dict()
    
    path = ''
    
    InitialBlankPathScore[path] = y[0]
    InitialPathsWithFinalBlank = set([path])

    
    InitialPathsWithFinalSymbol = set()
    for index, char in enumerate(SymbolSet):
        path = char
        InitialPathScore[path] = y[index+1]
        InitialPathsWithFinalSymbol.add(path)
        
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore
    
def ExtendWithBlank(PathsWithTerminalBlank,PathsWithTerminalSymbol, y) :

    
    UpdatedPathsWithTerminalBlank = set()
    UpdatedBlankPathScore = dict()

    for path in PathsWithTerminalBlank:
        UpdatedPathsWithTerminalBlank.add(path)

        UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]

    for path in PathsWithTerminalSymbol:
        
        if path in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[path] += PathScore[path] * y[0]
            
        else:
            UpdatedPathsWithTerminalBlank.add(path)
            UpdatedBlankPathScore[path] = PathScore[path] * y[0]
            
    return UpdatedPathsWithTerminalBlank,UpdatedBlankPathScore


 
def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y):
    UpdatedPathsWithTerminalSymbol = set()
    UpdatedPathScore = dict()
    
    for path in PathsWithTerminalBlank:
         for index, char in enumerate(SymbolSet): # SymbolSet does not include blanks
             newpath = path + char # Concatenation
             UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
             UpdatedPathScore[newpath] = BlankPathScore[path] * y[index+1]
             
    for path in PathsWithTerminalSymbol:
        # Extend the path with every symbol other than blank
        for index, c in enumerate(SymbolSet): # SymbolSet does not include blanks
            if c == path[-1]:
                newpath = path
            else:
                newpath = path + c
                
            if newpath in UpdatedPathsWithTerminalSymbol: # Already in list, merge paths
                UpdatedPathScore[newpath] += PathScore[path] * y[index+1]
            else: # Create new path
                UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
                UpdatedPathScore[newpath] = PathScore[path] * y[index+1]
                
    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore = dict()
    PrunedPathScore = dict()
    
    scorelist =[]

    for index, p in enumerate(PathsWithTerminalBlank):
        scorelist.append(BlankPathScore[p])
    
    for p in PathsWithTerminalSymbol:
        scorelist.append(PathScore[p])
    
    scorelist.sort(reverse=True)
    
    if BeamWidth < len(scorelist):
        cutoff = scorelist[BeamWidth-1]
    else:
        cutoff = scorelist[-1]
    
    PrunedPathsWithTerminalBlank = set()
    for p in PathsWithTerminalBlank:
        if BlankPathScore[p] >= cutoff:
             PrunedPathsWithTerminalBlank.add(p) # Set addition
             PrunedBlankPathScore[p] = BlankPathScore[p]
     
    PrunedPathsWithTerminalSymbol = set()
    for p in PathsWithTerminalSymbol:
        if PathScore[p] >= cutoff:
            PrunedPathsWithTerminalSymbol.add(p) # Set addition
            PrunedPathScore[p] = PathScore[p]
    
    return(PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore)
                         
def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore,PathsWithTerminalSymbol, PathScore):
    MergedPaths = PathsWithTerminalSymbol
    FinalPathScore = PathScore
    
    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] += BlankPathScore[p] 
        else:
            MergedPaths.add(p)
            FinalPathScore[p] = BlankPathScore[p]
    
    return MergedPaths, FinalPathScore
           


    