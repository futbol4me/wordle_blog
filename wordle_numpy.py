import numpy as np

def load_words():
    ''' loads words from text file; returns numpy array of strings
    ''' 
    with open('wordle_blog/five_letter_words.txt','r') as f:
        wordgen = (w for w in f.read().upper().split() if len(w)==5 and w.isalpha())
    
    return np.array(tuple(wordgen))


def mask_first(arr,target,replace):
    ''' helper method that masks characters already used up; if a letter is green, then dont 
        allow that letter to be double counted as a yellow
    '''
    t = np.where(arr==target)[0]
    if len(t):
        arr[np.where(arr==target)[0][0]]=replace


    
def trim(words, guess, colorcode,timed=False):
    ''' based on a guess and colorcode, returns a reduced wordlist
        that satisfies the conditions
    '''
    
    chars = words.view('<U1').reshape(words.size,-1)
    guess = np.array([c for c in guess])
    coded = np.array([c for c in colorcode])

    # check green rules,  chars[:,colfilter]==  is the numpy element wise comparison for each letter
    # then eliminate words using the boolean mask
    colfilter = (coded == 'G')
    mask = np.all(chars[:,colfilter]==guess[colfilter],axis=1)
    words = words[mask]
    chars = chars[mask]

    # must process non-greens consecutively
    colfilter = (coded != 'G')
    chars_xg = np.copy(chars)[:,colfilter]  # make a copy of non-green cols because we will mutate
    non_green_columns = zip(guess[colfilter],coded[colfilter],np.arange(sum(colfilter)))
    for (letter, code, pos) in non_green_columns:
        if code=='-' or code=='B':
            mask = np.all(chars_xg!=letter, axis=1)
            chars_xg = chars_xg [mask]
            words = words [mask] 

        else:  # deal with yellow
            # eliminate words with yellow letters in the same column
            mask = chars_xg[:,pos]!=letter
            chars_xg = chars_xg[mask]
            words = words[mask]

            # must find yellow in other columns
            othercols = tuple(set(np.arange(sum(colfilter)))-{pos})
            other_chars_xg = chars_xg[:,othercols]
            mask = np.any(other_chars_xg==letter, axis=1)
            np.apply_along_axis(mask_first,0,other_chars_xg,letter,"0")
            chars_xg = chars_xg[mask]
            words = words[mask]

    return words

def words_that_satisfy (wordlist, clue):
    return trim(wordlist, *clue)

def rate(words,guess):
    return sum((trim(words, guess, color_result(guess,word)).size) for word in words)/len(words)

def color_result(guess, answer):
    green = [True if guess[i]==answer[i] else False for i in range(5)]
    used = green[:]
    yellow = [False,False,False,False,False]
    for gu in range(5):
        for an in range(5):
            # print(' ',gu,guess[gu],an,answer[an],yellow,used)
            if gu!=an and not green[gu] and not(used[an]) and guess[gu]==answer[an]:
                yellow[gu]=True
                used[an]=True
                # print('*',gu,guess[gu],an,answer[an],yellow,used)
                break
    result =[0 for _ in range(5)]
    for i in range(5):
        if green[i]:
            result[i]="G"
        elif yellow[i]:
            result[i]="Y"
        else:
            result[i]="-"
    return "".join(result)

def get_color_clue(guess, answer):
    return [guess,color_result(guess,answer)]


def satisfies(colors,potentialAnswer):
    return color_result(colors[0],potentialAnswer)==colors[1]


words = load_words()

