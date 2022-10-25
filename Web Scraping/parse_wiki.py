import io
import wikipedia
import warnings
warnings.filterwarnings('ignore')


def parse_wiki(file_name, lang, words):
    wikipedia.set_lang(lang)

    try:
        with io.open(file_name, 'a', encoding='utf-8') as fout:
            print('Сбор данных начался! :)')
            for key_word in words:
                try:
                    titles = wikipedia.search(key_word)
                    for title in titles:
                        try:
                            summary = wikipedia.summary(title).strip()
                            fout.write(summary + '\n')
                        except:
                            continue
                except:
                    continue
        print(f'Данные собраны и записаны в файл "{file_name}".')

    except:
        print('Что-то пошло не так, скрипт не отработал (:')



lang = 'hi'
words = ['यूएसए']
file_name = 'output.txt'

parse_wiki(file_name, lang, words)