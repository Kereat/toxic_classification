from typing import List, Dict
import re
import itertools
from functools import partial
import string
import logging
import collections
from utils import LazyProperty

import numpy as np
import pandas as pd

# NLP
import nltk
from textacy import preprocess
from razdel import tokenize
from nltk.corpus import stopwords
import pymorphy2
from pymystem3 import Mystem
from natasha import NamesExtractor
from alphabet_detector import AlphabetDetector

import word_lists

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(name)s %(funcName)s %(message)s')

# Custom types
Tokenlist = List[str]


class PreprocessingInterface(object):
    def __init__(self):
        self.mystem = Mystem()
        self.names_extractor = NamesExtractor()
        self.pymorphy = pymorphy2.MorphAnalyzer()
        self.alphabet_detector = AlphabetDetector()
        
        self.stop_words = set(word_lists.yandex_seo_stopwords + stopwords.words('russian'))
        
        self.padding_punct = """'`"""
        self.full_punct = string.punctuation + '«-–»'
        
        self.line_break_trans = str.maketrans('\r\t.', '\n\n\n')
        
        self.unwanted_punct = ",.:!?0#№«»()-\"'_="
        self.unwanted_punct_trans = str.maketrans(self.unwanted_punct, ' '*len(self.unwanted_punct))
        
        self.pipeline = [
            preprocess.fix_bad_unicode,
            preprocess.replace_urls,
            preprocess.replace_emails,
            preprocess.replace_phone_numbers,
            preprocess.replace_numbers,
            preprocess.replace_currency_symbols,
            preprocess.remove_accents,
            
            partial(preprocess.remove_punct, marks=""",.:;!?0%@#№`«»<>()+[]-\/'"_="""),
            preprocess.normalize_whitespace,
            nltk.word_tokenize,
            lambda x: " ".join(x)
            # pp.replace_digits
        ]
        
    # ======================================== #
    # ########### MAIL PROCESSING ############ #
    # ======================================== #
    def is_forwarding(self, paragraph: str) -> bool:
        """ Checks if line contains forwarding message markup """
        return any([paragraph.startswith(each) for each in word_lists.forward_markup])
    
    def is_mail_form(self, paragraph: str) -> bool:
        """ Checks if line contains mail forms """
        return any([paragraph.startswith(each) for each in word_lists.mail_forms])
    
    def filter_mail_lines(self, raw: str):
        """ Line breaks need to be normalized first
        """
        paragraphs = raw.split('\n') # TODO: rewrite with sent tokenizer
        lines = [line for line in paragraphs if not self.is_mail_form(line)]
        text = ' '.join(lines)
        text = text.replace(
            'добрый день', '').replace(
            'здравствуйте', '').replace(
            'спасибо', '').replace(
            'хорошего дня', '').replace(
            'subject', '').replace(
            'заявка', '')
        return text

    def split_paragraph(raw: str):
        normalized_string = PreprocessingInterface.normalize(raw)  # replacing ("\r", "\t", ".") with "\n"
        split_strings = re.split('\n', normalized_string)
        filtering = list(filter(lambda x: x if x != " " else None, split_strings))
        filtered = [x.strip() for x in filtering]
        return filtered

    @LazyProperty
    def re_intros(self):
        return [re.compile(each) for each in word_lists.intros]

    @LazyProperty
    def re_signatures(self):
        return [re.compile(each) for each in word_lists.signatures]

    def cut_mail_intros(self, raw: str) -> str:
        """ Cut by input search pattern """
        # 1. Split
        # 2. Tag each split
        # 3. Decide where to slice
        # for each in self.re_intros:
            # each.match()
        pass
    
    def cut_mail_signatures(self, raw: str) -> str:
        """
        Find index of first signature using precompiled regex
        and cut it
        """
        pass
        
    def cut_by_signature(raw: str, signature: str) -> str:
        """ Cut by input search pattern """
        p = re.compile(signature)
        search = p.search(raw)
        try:
            start_index = search.span()[0]  # start index
            if start_index > 4:  # Do not cut from the beginning
                return raw[:start_index]
            else: 
                return raw
        except AttributeError:  # Not found
            return raw

    def cut_signatures(self, raw: str) -> str:
        """
        Find index of first signature using precompiled regex
        and cut it
        """
        beginnings = []
        for each in self.re_signatures:
            try:
                # Index of 1st found position
                beginnings.append(each.search(raw).span()[0])
            except AttributeError:
                pass

        if beginnings:
            cut = min(beginnings)
            # Not in the beginning
            if cut > 5:
                return raw[:cut]
            else:
                return raw
        return raw
        
    # ======================================== #
    # ############# STRING LEVEL ############# #
    # ======================================== #
    def preprocessing_pipeline(self, raw: str, pipeline=None):
        """pipeline: List of prepricessing funcs like """
        if not pipeline:
            pipeline = self.pipeline
        for fn in pipeline:
            raw = fn(raw)
        return raw

    def pad_punctuation(self, raw: str, punct_list=None) -> str:
        """
        Adds whitespaces before and after each punctuation symbol
        Used to control tokenization
        """
        normal_text = raw.strip()
        padding_punctuation = punct_list if punct_list else self.padding_punct
        for char in padding_punctuation:
            normal_text = normal_text.replace(char, ' ' + char + ' ')
        return normal_text
    
    @staticmethod
    def razdel_tokenize(raw: str):
        return [each.text for each in list(tokenize(raw))]

    def is_cyrillic(self, token) -> bool:
        """ 
        Checks if string has only cyrillic letters
        """
        if self.contains_digits(token) or self.contains_punct(token):
            return False
        else:
            return self.alphabet_detector.only_alphabet_chars(token, 'CYRILLIC')
    # ======================================== #
    # ############## TOKEN LEVEL ############# #
    # ======================================== #
    @staticmethod
    def replace_digits(tokens: list):
        return [t if not t.isdigit() else "*digit*" for t in tokens]
    
    # ======================================== #
    # ########### POS/LEMMATIZING ############ #
    # ======================================== #
    def lemmatize_with_mystem(self, raw: str):
        if isinstance(raw, list):
            raw = " ".join(raw)
        lemmatized_tokens = self.mystem.lemmatize(raw)
        lemmas_filtered = [t for t in lemmatized_tokens if t != ' ' and t != '\n']  # filter empty
        if len(lemmas_filtered) == 0:
            return ""
        return " ".join(lemmas_filtered).strip()
    
    def get_pymorphy_lemma(self, token: str) -> str:
        return self.pymorphy.parse(token)[0].normal_form

    def lemmatize_tokens_with_mystem(self, text: Tokenlist) -> Tokenlist:
        lemmatized_tokens = self.mystem.lemmatize(" ".join(text))
        lemmas_filtered = [t for t in lemmatized_tokens if t != ' ' and t != '\n']  # filter empty
        return lemmas_filtered

    def lemmatize_with_pymorphy(self, text: Tokenlist) -> Tokenlist:
        lemmas = []
        for token in text:
            p = self.pymorphy.parse(token)[0]
            lemmas.append(p.normal_form)
        lemmas_filtered = [t for t in lemmas if t != ' ' and t != '\n']  # filter empty
        return lemmas_filtered

    def get_mystem_pos(self, token):  # TODO: apply mystem to whole text
        response = self.mystem.analyze(token)
        analysis = response[0].get('analysis')
        try:
            the_one = analysis[0]
            tag = the_one.get('gr')
            return tag
        except Exception as e:
            print(e, e.args)
            return None
        
    # ======================================== #
    # ######### Mail froms cleaning ########## #
    # ======================================== #    
    
    def parse_mystem_tags(self, analysis):
        if analysis.get("analysis"):
            if "gr" in analysis["analysis"][0]:
                tag_string = analysis["analysis"][0]["gr"]
                result = tag_string.split(",")
                return result
        return ""
    
    def mystem_remove_after_fio(self, raw):
        analysis = self.mystem.analyze(raw)
        result = []
        for i, each in enumerate(analysis):
            result.append(each["text"])
            if "фам" in self.parse_mystem_tags(each):
                if i<len(analysis)-3:
                    if "имя" in self.parse_mystem_tags(analysis[i+2]) and len(result)>3:
                        return "".join(result[:-1]).strip('\n')
            elif "имя" in self.parse_mystem_tags(each):
                if i<len(analysis)-3:
                    if "отч" in self.parse_mystem_tags(analysis[i+2]) and len(result)>3:
                        return "".join(result[:-1])
        return "".join(result).strip('\n')

    def natasha_remove_after_fio(self, raw):
        matches = self.names_extractor(raw)
        if matches:
            for each in matches:
                name = each.fact.first
                patr = each.fact.middle
                surname = each.fact.last
                cut_index = each.span[0]
                if name and  surname or name and patr and cut_index > 30:
                    return raw[:each.span[0]]
                else: 
                    return raw
        else: 
            return raw
    
    # ======================================== #
    # ############## Filtering ############### #
    # ======================================== #
    
    def get_vocab(self, tokenized_texts: pd.Series) -> set:
        return set(self.series_to_chain(tokenized_texts))

        
    def remove_stopwords(self, text: Tokenlist, stopwords: list=None) -> Tokenlist:
        if not stopwords:
            stopwords = self.stop_words
        return [t for t in text if t not in stopwords]    

    @staticmethod
    def filter_by_token_length(text: Tokenlist, min=1, max=25) -> Tokenlist:
        return [t for t in text if len(t) >= min and len(t) <= max]
                
    """
    def mystem_remove_names(self, text: Tokenlist):
        result = []
        for each in self.mystem.analyze(" ".join(text)):
            if not each['text'] in (" ", "\n"):
                if 'имя' not in tags and 'фам' not in parse_mystem_tags(each)
                    result.append(each["text"])
        return result
    """
                
    def pymorphy_isname(self, token: str):
        """ Better then mystem? """
        tags = self.pymorphy.parse(token)[0].tag
        if 'Name' in tags or 'Surn' in tags or 'Patr' in tags:
            return True
        else:
            return False

    def pymorphy_remove_names(self, text: Tokenlist):
        """ Takes pymorphy_isname/ as input"""
        return [t for t in text if not self.pymorphy_isname(t) or t in word_lists.pymorphy_name_exceptions]

    def get_names_df(self, df_col, name_extractor):
        ctr = collections.Counter(list(self.series_to_chain(df_col)))
        fdist_list = ctr.most_common()
        res = {each[0]: each[1] for each in fdist_list if name_extractor(each[0])}
        df = pd.DataFrame.from_dict(res, orient='index')
        df.columns = ["count"]
        df["token"] = df.index
        df.index = [list(range(len(df)))]
        return df
                
    # ======================================== #
    # ########### Pandas analysis ############ #
    # ======================================== #

    def get_nltk_pos_df(self, texts: pd.Series) -> pd.DataFrame:
        all_tokens = self.series_to_chain(texts)
        nltk_tags_tuple = nltk.pos_tag(all_tokens, lang='rus')
        tags = set([each[1] for each in nltk_tags_tuple])

        def get_tokens_by_tag(tag):
            # Set of tokens by input tag
            token_tag_list = list(filter(lambda x: x[1] == tag, nltk_tags_tuple))  # [token, tag]
            return [each[0] for each in token_tag_list]  # [token]

        tag_dict = collections.OrderedDict(zip(tags, [get_tokens_by_tag(tag) for tag in tags]))
        return pd.DataFrame.from_dict(tag_dict, orient='index').transpose()

    def get_mystem_pos_df(self, texts: pd.Series) -> pd.DataFrame:
        all_tokens = self.series_to_chain(texts)
        mystem_tags_dict = {token: self.get_mystem_pos(token) for token in set(all_tokens)}
        # filter_dict(mystem_tags_dict)
        mystem_tags_dict = dict(filter(lambda item: item[1] is not None, mystem_tags_dict.items()))      
        
        def get_tokens_by_mystem_tag(input_tag):
            matched_tokens = [(token, all_tokens.count(token)) for token, tags in mystem_tags_dict.items() if input_tag in tags]
            return sorted(matched_tokens, key=lambda x: x[1], reverse=True)
        # {tag: (token, count), ...}
        mystem_tag_dict = collections.OrderedDict(zip(word_lists.forbidden_mystem_tags,
                                                      [get_tokens_by_mystem_tag(tag) for tag in
                                                       word_lists.forbidden_mystem_tags]))
        return pd.DataFrame.from_dict(mystem_tag_dict, orient='index').transpose()

    # ======================================== #
    # ########## Search/stats ############ #
    # ======================================== #
    @staticmethod
    def stats_for_untokenized(texts: pd.Series):
        """ Counts symbols in series of texts """
        return sum([len(each) for each in texts])

    @staticmethod
    def series_to_chain(texts: pd.Series) -> Tokenlist:
        """ Chained tokens in Series """
        return list(itertools.chain.from_iterable(list(texts.values)))

    def stats_for_series(self, texts: pd.Series) -> pd.DataFrame:
        """DF from Series stats"""
        empty_texts_indexes = list(texts[texts.astype(str) == '[]'].index)
        empty_texts = len(empty_texts_indexes)
        token_chain = self.series_to_chain(texts)

        result = pd.DataFrame(data=[
            [len(token_chain),
             len(list(set(token_chain))),
             len(texts),
             empty_texts,
             token_chain.count('')]
        ],
            index=['Count'],
            columns=['Total tokens',
                     'Unique tokens',
                     'Total texts',
                     'Empty texts',
                     'Empty tokens'])
        return result

    @staticmethod
    def check_empty_texts(texts: pd.Series, original_df=None):
        """
        Get unprocessed text for '[]' in Series
        :returns list of indexes or pd.Index
        """
        empty_texts_indexes = list(texts[texts.astype(str) == '[]'].index)
        if original_df:
            return original_df.loc[empty_texts_indexes]
        else:
            return empty_texts_indexes

    @staticmethod
    def plot_occurrences(data: pd.Series, expression):
        """
        Detects first occurrence of str expression in text.
        Plots index distribution of occurrences.
        """
        indexes = [text.index(expression) for text in data if expression in text]
        fig, ax = plt.subplots()
        ax.hist(indexes, range(0, 50))
        ax.set_xticks(np.arange(0, 51, 1))
        ax.set_xlabel('Position')
        ax.set_ylabel('Count')
        plt.title("Occurrence distribution")
        print(len(indexes), ' occurrences found')
        return ax

    def get_token_counts_df(self, texts: pd.Series, topn=30) -> pd.DataFrame:
        ctr = collections.Counter(list(self.series_to_chain(texts)))
        fdist_list = ctr.most_common(topn)
        tokens = [k for k, v in fdist_list]
        counts = [v for k, v in fdist_list]
        return pd.DataFrame({"token": tokens, "count": counts})

    def plot_token_frequencies(self, texts: pd.Series, topn=30) -> sns.barplot():
        """ Plot frequency distribution over corpus for top_n tokens tokens """
        get_token_counts_df = self.get_token_counts_df(texts, topn)
        sns.barplot(x="count", y="token", data=get_token_counts_df).set_xlabel('Token appearence')

    def plot_token_distribution(self, texts: pd.Series):
        """ Overall tokens lenghts distribution for series """
        token_lenghts = [len(x) for x in self.series_to_chain(texts)]
        bow_lenghts = [len(x) for x in texts]

        # Unique lens
        fig, ax = plt.subplots(ncols=2)

        ax[0].hist(token_lenghts, bins=range(0, 25))
        ax[0].set_xticks(np.arange(0, 26, 1))
        ax[0].set_xlabel('Token length')
        ax[0].set_ylabel('Count')

        ax[1].hist(bow_lenghts, bins=range(0, 25))
        ax[1].set_xticks(np.arange(0, 26, 1))
        ax[1].set_xlabel('Tokens in text')
        ax[1].set_ylabel('Count')
        return ax

    @staticmethod
    def get_most_common(data: pd.DataFrame) -> pd.DataFrame:
        # df = self.get_categories_df(series)
        result = dict()
        for col in data.columns:
            try:
                col_most_freq = data[col].value_counts().reset_index()
                tokens = col_most_freq['index']
                freqs = col_most_freq[col]
                result[col] = [(t, f) for t, f in zip(tokens, freqs)]
            except:
                result[col] = [None]
        return pd.DataFrame.from_dict(result, orient='index').transpose()

    # ======================================== #
    # ###### TOKEN SEQUENCE PROCESSING ####### #
    # ======================================== #
    @staticmethod
    def get_texts_with_token(texts: pd.Series, token: str) -> List[Tokenlist]:
        return [text for text in texts if token in text] # TODO: Refactor with pandas

    @staticmethod
    def cut_after_token(text: Tokenlist, token: str, shift=0) -> Tokenlist:
        """ Truncate list after input token + index shift (optional) """
        if token in text:
            if text.index(token) > 1:
                return text[:text.index(token) + shift]
            else:
                return text
        else:
            return text

    @staticmethod
    def get_indexes_of_token(texts: pd.Series, token: str) -> List:
        """ Indexes of the token in all documents """
        indexes = [text.index(token) for text in texts if token in text]
        return indexes

    @staticmethod
    def token_scope(texts: pd.Series, token: str, pos) -> Tokenlist:
        """ Set of tokens going before or after (by position) the given token """
        found = texts.apply(lambda x: x[x.index(token) + pos] if token in x else 0)
        token_set = list(set(found[found != 0]))
        return token_set

    @staticmethod
    def seq_in_series(texts: pd.Series, seq: List) -> Tokenlist:
        """ Return texts in which sequence present """
        result = []
        for text in texts:
            if seq[0] in text:
                index = text.index(seq[0])
                if seq == text[index:(index + len(seq))]:
                    result.append(text)
        return result

    def plot_indexes_of_token(self, texts: pd.Series, token: str, x_range):
        indexes = self.get_indexes_of_token(texts, token)
        fig, ax = plt.subplots()
        ax.hist(indexes, bins=range(0, x_range))
        ax.set_xticks(np.arange(0, x_range + 1, 1))
        ax.set_yticks(np.arange(0, 21, 1))
        ax.set_xlabel('Index')
        ax.set_ylabel('Count')
        plt.title(token)
        return ax

    @staticmethod
    def cut_after_seq(text: Tokenlist, seq: Tokenlist) -> Tokenlist:
        """ Truncate document after token sequence """
        if seq[0] in text: # if first element of seq is in text
            index = text.index(seq[0])
            if seq == text[index:(index + len(seq))]:  # if whole sequence is is
                return text[:text.index(seq[0])]
            else:
                return text
        else:
            return text

    @staticmethod
    def cut_seq(text: Tokenlist, seq: Tokenlist) -> Tokenlist:
        """ Removes sequence from tokenized texts. """
        if seq[0] in text:
            index = text.index(seq[0])
            if seq == text[index:(index + len(seq))]:
                '''
                for each in seq:
                    del tokenlist[tokenlist.index(each)]
                return tokenlist
                '''
                return text[:index] + text[index + len(seq):] # TODO: test it
            else:
                return text
        return text

    # ======================================== #
    # ################ OTHER ################# #
    # ======================================== #
    def separate_by_category(self, texts: pd.Series) -> Dict:
        """
        Separates tokens by types of chars in it (punctuation, numbers, ...)
        :param texts: series of tokenized texts
        :return: dict of {category:[tokenlist]}
        """
        vocab = self.series_to_chain(texts)

        result = {'num_punct': [],
                  'alpha_num': [],
                  'alpha_punct': [],
                  'punct_tokens': [],
                  'numeric_tokens': [],
                  'alpha_tokens': [],
                  'alpha_num_punct': []}

        for token in vocab:
            # Add flag by symbol category
            punct = [1 for symbol in token if (symbol in self.full_punct)]
            numerics = [1 for symbol in token if (symbol.isnumeric())]
            alpha = [1 for symbol in token if (symbol.isalpha())]

            # If token contains all types
            if (punct and numerics) and alpha:
                result['alpha_num_punct'].append(token)

            # Double
            elif numerics and punct:
                result['num_punct'].append(token)

            elif numerics and alpha:
                result['alpha_num'].append(token)

            elif alpha and punct:
                result['alpha_punct'].append(token)

            # Simple
            elif punct:
                result['punct_tokens'].append(token)

            elif numerics:
                result['numeric_tokens'].append(token)

            elif alpha:
                result['alpha_tokens'].append(token)

        return result

    def get_categories_df(self, texts: pd.Series) -> pd.DataFrame:
        # make df from separation dict
        separated_categories_dict = self.separate_by_category(texts)
        categories = pd.DataFrame.from_dict(separated_categories_dict, orient='index')
        return categories.transpose()

    @staticmethod
    def fix_year(date: str) -> int:
        year = date[:4]
        year = int(year) - 2000
        return year
    
    # ======================================== #
    # ############## PIPELINES ############### #
    # ======================================== #

    @staticmethod
    def merge_ticket_fields(subject, description):
        """ Removes theme copy from description, """
        if not isinstance(subject, str):
            subject = ""
        if not isinstance(description, str):
            description = ""
        if description.startswith(subject):
            return description
        else:
            return "{} {}".format(subject, description).strip()

    def apply_pipeline(self, raw: str) -> Tokenlist:
        """ Apply all the methods to raw string """
        normalized = self.normalize(raw)
        padded = self.pad_punctuation(normalized)
        tokenized = self.tokenize(padded)
        no_punct = self.remove_punct(tokenized)
        no_stops = self.remove_stopwords(no_punct)
        cut_by_len = [t for t in no_stops if len(t) < 25]
        lemmatized = self.lemmatize_tokens_with_mystem(cut_by_len)
        return lemmatized

    def apply_inference_pipeline(self, subject, description: str) -> Tokenlist:
        """ Preprocessing for model inference in production """
        merged_text = self.merge_ticket_fields(subject, description)
        punct_padded = self.pad_punctuation(merged_text)
        normalized = self.normalize(punct_padded)
        tokenized = self.tokenize(normalized)
        return tokenized
