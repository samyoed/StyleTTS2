from gruut import sentences
from collections.abc import Iterable
import phonemizer
from nltk.tokenize import word_tokenize

espeak_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)



class PhonemeConverter:
    def phonemize(self, text):
        pass


class GruutPhonemizer(PhonemeConverter):
    def phonemize(self, text, lang='en-us'):
        phonemized = []
        for sent in sentences(text, lang=lang):
            for word in sent:
                if isinstance(word.phonemes, Iterable):
                    phonemized.append(''.join(word.phonemes))
                elif isinstance(word.phonemes, str):
                    phonemized.append(word.phonemes)
        phonemized_text = ' '.join(phonemized)
        return phonemized_text


class ESpeakPhonemizer(PhonemeConverter):
    def phonemize(self, text, lang='en-us'):
        text = text.strip()
        text = text.replace('"', '')
        ps = espeak_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        return ps


class PhonemeConverterFactory:
    @staticmethod
    def load_phoneme_converter(name: str, **kwargs):
        if name == 'gruut':
            return GruutPhonemizer()
        if name == 'espeak':
            return ESpeakPhonemizer()
        else:
            raise ValueError("Invalid phoneme converter.")