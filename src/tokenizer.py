class Tokenizer:
    def __init__(self):
        return

    def compute_vocab(self, text):
        self.vocab = list(set(text))
        self.vocab.sort()
        self.ctoi = { self.vocab[i]:i for i in range(len(self.vocab)) }
        self.itoc = { i:self.vocab[i] for i in range(len(self.vocab)) }

    def get_vocab_size(self):
        return len(self.vocab)

    def encode(self, s): 
        return [ self.ctoi[i] for i in s ]

    def decode(self, tokens): 
        return "".join([ self.itoc[i] for i in tokens ])

###############################

import bisect

def find_longest_token(s, vocab):
    last = -1
    for fin in range(0, len(s)):
        ss = s[:fin+1]
        #print("**", ss, "**")
        f = bisect.bisect_left(vocab, ss)
        if f>=len(vocab):
            break
        elif (vocab[f]==ss):            
            last=f
        else:
            break

    assert last!=-1
    return last

class Tokenizer2:
    def __init__(self):
        self.vocab = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'th', 'ou', 'an', 'er', 'in', 'the', 'or', 'en', 'ar', 'is', 'on', 'es', 'at', 'and', 'll', 'to', 'st', 'you', 'me', 'no', 'ha', 'ing', 'of', 'se', 'he', 'le', 'be', 'wi', 'wh', 're', 'it', 've', 'ro', 'my', 'ch', 'for', 'ce', 'as', 'ay', 'that', 'li', 'ed', 'ir', 'we', 'ld', 'ere', 'us', 'ut', 'ke', 'not', 'ri', 'de', 'lo', 'with', 'so', 'gh', 'your', 'ent', 'co', 'hi', 'thou', 'our', 'go', 'al', 'ow', 'ad', 'his', 'but', 'et', 'un', 'this', 'ther', 'est', 'ard', 'all', 'have', 'ly', 'ur', 'do', 'king', 'ght', 'ra', 'him', 'ord', 'od', 'ma', 'pe', 'ess', 'what', 'now', 'am', 'thy', 'ti', 'ver', 'sha', 'fe', 'ge', 'will', 'ould', 'are', 'ck', 'id', 'man', 'one', 'ue', 'fa', 'by', 'ne', 'io', 'la', 'her', 'pr', 'su', 'po', 'con', 'lord', 'if', 'ter', 'ta', 'shall', 'end', 'sh', 'mor', 'up', 'whi', 'art', 'thee', 'tr', 'com', 'bo', 'let', 'ho', 'come', 'ence', 'mo', 'du', 'ca', 'fro', 'good', 'ous', 'ius', 'oun', 'ain', 'sel', 'there', 'ry', 'mar', 'mu', 'si', 'from', 'te', 'well', 'she', 'then', 'was', 'who', 'than', 'ul', 'here', 'pro', 'more', 'which', 'ine', 'ven', 'bro', 'would', 'ill', 'ward', 'per', 'sir', 'ion', 'can', 'um', 'ble', 'men', 'they', 'ath', 'ight', 'ong', 'rich', 'them', 'out', 'ant', 'sa', 'row', 'say', 'duke', 'self', 'their', 'ct', 'wor', 'how', 'see', 'ak', 'know', 'love', 'gl', 'war', 'ate', 'ind', 'gi', 'ag', 'vi', 'hath', 'like', 'wo', 'may', 'were', 'vo', 'spe', 'when', '--', 'where', 'que', 'qu', 'fo', 'richard', 'nor', 'fri', 'gre', 'fore', 'rome', 'queen', 'ju', 'ers', 'di', 'ii', 'fir', 'ep', 'upon', 'some', 'ish', 'ok', 'gra', 'make', 'had', 'yet', 'must', 'should', 'ves', 'way', 'ci', 'pa', 'ple', 'ink', 'edward', 'fi', 'first', 'son', 'did', 'own', 'br', 'ear', 'ady', 'god', 'blo', 'ound', 'il', 'dis', 'cor', 'thing', 'der', 'death', 'au', 'ure', 'hen', 'ty', 'uc', 'ity', 'tis', 'again', 'too', 'why', 'hast', 'use', 'henry', 'speak', 'ince', 'day', 'el', 'ork', 'york', 'take', 'romeo', 'lan', 'time', 'ba', 'give', 'ester', 'gent', 'father', 'mer', 'ser', 'cl', 'word', 'heart', 'ouc', 'think', 'ex', 'ange', 'hea', 'hon', 'ful', 'glouc', 'abe', 'gloucester', 'mis', 'mon', 'hand', 'these', 'ance', 'swe', 'less', 'look', 'gentle', 'most', 'lady', 'tell', 'sp', 'brother', 'mine', 'inc', 'ab', 'jo', 'very', 'blood', 'hear', 'such', 'des', 'wick', 'bre', 'land', 'ber', 'na', 'str', 'coun', 'warwick', 'other', 'af', 'friend', 'gn', 'lt', 'ac', 'ort', 'entio', 'stand', 'llow', 'ey', 'ness', 'life', 'urse', 'ap', 'iz', 'fear', 'fair', 'oul', 'ving', 'honour', 'any', 'made', 'new', 'ling', 'pre', 'off', 'night', 'inius', 'im', 'tion', 'lea', 'lie', 'ans', 'much', 'corio', 'ved', 'pl', 'ts', 'ang', 'great', 'ise', 'mi', 'vinc', 'urn', 'vincentio', 'ings', 'pray', 'ment', 'call', 'pla', 'heaven', 'ster', 'true', 'part', 'cond', 'chi', 'wn', 'bear', 'never', 'gr', 'tle', 'done', 'prince', 'gar', 'ation', 'ford', 'lanus', 'noble', 'coriolanus', 'ef', 'cap', 'wer', 'cannot', 'doth', 'juli', 'sweet', 'set', 'menen', 'menenius', 'ham', 'age', 'arm', 'juliet', 'before', 'both', 'head', 'rown', 'fort', 'second', 'bet', 'name', 'thus', 'grace', 'red', 'fl', 'par', 'being', 'ough', 'bl', 'int', 'gu', 'been', 'hold', 'cut', 'broke', 'away', 'wr', 'tho', 'ces', 'live', 'pp', 'ru', 'res', 'soul', 'though', 'dead', 'isabe', 'put', 'angelo', 'cour', 'ol', 'down', 'pt', 'ont', 'against', 'ree', 'ase', 'cit', 'luc', 'poor', 'ouse', 'deed', 'tru', 'long', 'ret', 'leave', 'pet', 'ever', 'nurse', 'clar', 'mad', 'clarence', 'myself', 'ars', 'ies', 'pres', 'clo', 'mother', 'ast', 'ace', 'pri', 'thought', 'op', 'ice', 'stay', 'rest', 'ign', 'ite', 'comes', 'world', 'therefore', 'ss', 'izen', 'cal', 'citizen', 'child', 'till', 'even', 'boling', 'bolingbroke', 'ass', 'cause', 'lla', 'friends', 'eyes']
        self.vocab.sort()

    def compute_vocab(self, text):
        return

    def get_vocab_size(self):
        return len(self.vocab)

    def encode(self, s): 
        a = zip(range(len(self.vocab)), self.vocab)
        v_id,v_vo = zip(*list(sorted(a, key=lambda a: a[1])))

        tokens = []
        i = 0
        while i<len(s):
            t = find_longest_token(s[i:], v_vo)
            tokens.append(v_id[t])
            i+= len(v_vo[t])
        return tokens        
    def decode(self, tokens): 
        return "".join([ self.vocab[i] for i in tokens ])