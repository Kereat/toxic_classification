import re
import pymorphy2
from natasha import NamesExtractor

morph = pymorphy2.MorphAnalyzer()
extractor = NamesExtractor()

class Criteria:
    @staticmethod
    def is_meaningful(text_list:list):
        #it is possible to futher limit this slice by eleminating improper grammems
        #however, it is rather impractical to do here - no notable effect was found
        length = 3
        
        for word in text_list:
            pp = morph.parse(word)[0]
            if {'NUMB'} in pp.tag:
                length+=1
                
        if len(text_list)<=length:
            return '!short'
        
    @staticmethod
    def is_mail_addr(text_list:list):
        at, mail = False, False
        for word in text_list:
            if '@' in word and '.' in word:
                return '!mail_addr'
            if '@' in word:
                at = True
            if 'mail' in word:
                mail = True
        if at+mail == 2:
            return '!mail_addr'
        if at+mail == 1:
            return '?mail_addr'
        
    @staticmethod
    def is_tel(text_list:list):
        number, telephone = False, False
        for word in text_list:
            if word.count('0')>7:
                telephone = True
            if 'тел.' in word.lower() or 'тел' == word[:3].lower():
                number = True

        if number+telephone == 2:
            return '!tele_num'
        elif number+telephone == 1:
            return '?tele_num'

    @staticmethod    
    def is_name(text_list:list):
        matches = extractor(' '.join(text_list))
        if len(matches)>0:
            return '?name'
        
    @staticmethod
    def is_loc(text_list:list):
        for word in text_list:
            pp = morph.parse(word)
            for x in pp:
                if {'NOUN', 'Geox'} in x.tag and x.score>0.4:
                    if word.isupper() is True:
                        return '?loc'
                    return '!loc'
    @staticmethod            
    def get_criteria_list():
        return [
            Criteria.is_loc,
            Criteria.is_mail_addr,
            Criteria.is_meaningful,
            Criteria.is_name,
            Criteria.is_tel
        ]

class SignatureRemover():
    def __init__(self, mode='strict',divide_by='\n'):
        self.mode = mode
        self.divider = divide_by
        self.error_counter = 0
    
    def remove(self, text):
        clean_text = self.clean(text)
        paragraph_list = self.split(clean_text)
        scheme = self.get_scheme(paragraph_list)
        a,b = self.get_culling_points(scheme)
        list_content = self.split(clean_text, None)[a:b]
        #print(list_content)
        concat_list = sum(list_content, [])
        return " ".join(concat_list)

    def clean(self,text):
        char_rep = str.maketrans('!?;:-,\r0123456789','....   0000000000')
        char_trunc = str.maketrans('', '', '(){}[]/-')
        text = text.replace('\n',' \n ').replace('\t', ' \t ')
        text = text.translate(char_rep)
        text = text.translate(char_trunc)
        return text.replace('0 0','00')

    def split(self, text, slc=5):
        text = self.clean(text)
        if self.divider == '\n':
            txt_list = [x for x in text.replace('.',' ').split('\n') if len(x)>5]
        elif self.divider == '.' :
            txt_list = [x for x in text.replace('\n',' ').split('.') if len(x)>5]
        elif type(self.divider) is list:
            txt_list = [x for x in re.split(', '.join(self.divider)) if len(x)>5]

        txt_list = [txt.split()[:slc] for txt in txt_list]
        return txt_list

    def get_scheme(self, txt_list):
        scheme = []
        for txt in txt_list:
            criteria = []
            for func in Criteria.get_criteria_list():
                criteria.append(func(txt))
            scheme.append(set(criteria)-{None})
            print(scheme[-1])
        return scheme

    def check_criteria(self,scheme_part, threshold):
        score = 0
        for criteria in scheme_part:
            if '!' in criteria:
                score+=2
            if '?' in criteria:
                score+=1
        return score
        
    def get_culling_points(self, scheme):
        if self.mode == 'strict':
            threshold = 2
        else:
            threshold = 3
            
        score_map = [self.check_criteria(x,threshold) for x in scheme]
        print(score_map)
        for x in range(1,len(score_map)-2):
            if score_map[x-1]+score_map[x+1]==0:
                score_map[x]=0
            if score_map[x]>0 and score_map[x+1]>=threshold and score_map[x+2]>=threshold:
                score_map[x]=threshold
        print(score_map)
        score_map = [int(x>=threshold) for x in score_map]
        print(score_map)
        try:
            a = score_map.index(0)
        except ValueError as e:
            print('Faulty input, every line is over specified threshold')
            raise e
        
        try:
            b = score_map.index(1,a)
        except ValueError:
            b = None
            
        return a,b