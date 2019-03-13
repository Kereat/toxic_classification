import logging
from pymystem3 import Mystem

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(name)s %(funcName)s %(message)s')

class PreprocessingAdapter(object):
    unwanted = ",.:!?0#№«»()-\"'_="
    punct = str.maketrans(unwanted, ''.join([' ' for x in range(len(unwanted))]))

    def __init__(self):
        self.mystem = Mystem()

    def preprocess(self, subject: str, description: str) -> str:
        raw = self.merge_description(subject, description)
        cleaned = self.makeshift_clean(raw)
        # lemmas = self.lemmatize_with_mystem(cleaned)
        return " ".join(lemmas)

    def merge_description(self, subject=" ", description=" ") -> str:
        if description.startswith(subject):
            return description[len(subject):].strip()
        else:
            return subject.strip() + " \n " + description.strip()

    def lemmatize_with_mystem(self, text: str) -> list:
        lemmatized_tokens = self.mystem.lemmatize(text)
        lemmas_filtered = [t for t in lemmatized_tokens if t != ' ' and t != '\n']  # filter empty
        if len(lemmas_filtered)==0:
            logger.warning("Empty description after lemmatization: {}".format(lemmas_filtered))
        return lemmas_filtered

    def is_forwarding(self, raw: str) -> bool:
        raw = raw.strip().lower()
        if raw.startswith('from:') or raw.startswith('sent:') or raw.startswith('to:') or raw.startswith('cc:') or raw.startswith('fw:'):
            return True
        return False

    def makeshift_clean(self, raw: str) -> str:
        raw = raw.replace('\xa0', ' ')
        # txt_list = raw.split('\n')
        # raw = ' '.join([x for x in txt_list if not self.is_forwarding(x)]).lower()
        raw = raw.replace(
            'добрый день',' ').replace(
            'здравствуйте',' ').replace(
            'спасибо',' ').replace(
            'хорошего дня',' ').replace(
            'subject',' ').replace(
            'заявка',' ').translate(self.punct)
        return ' '.join(raw.split())