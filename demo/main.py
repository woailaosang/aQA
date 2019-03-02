# -*- coding:utf-8 -*-
import web
from web import form
import sys, os
from mrc import get_an_example, any_input

os.system('cd /home/sangzhijie && bash logbuptgw.sh -i 2016110858 6361311')

# Import required modules.
from azure.cognitiveservices.search.websearch import WebSearchAPI
from azure.cognitiveservices.search.websearch.models import SafeSearch
from msrest.authentication import CognitiveServicesCredentials
def get_passage(question):
    passage_list = []
    try:
        subscription_key = "7c06d657db2e46c4b4c1c3b65bb45fa1"
        client = WebSearchAPI(CognitiveServicesCredentials(subscription_key))
        web_data = client.web.search(query=question, set_lang='EN')
        if hasattr(web_data.web_pages, 'value'):
            data_list = [item for item in web_data.web_pages.value if item.additional_properties['language'] == 'en']
            print("Total num: " + str(len(web_data.web_pages.value)) + '\tFiltered num: ' + str(len(data_list)))
            for i in range(len(data_list)):
                web_page = data_list[i]
                name = web_page.name
                url = web_page.url
                snippet = web_page.snippet
                passage_list.append([snippet, url])
        else:
            print("Didn't find any web pages...")
    except:
        os.system('cd /home/sangzhijie && bash logbuptgw.sh -i 2016110858 6361311')
        print("get_passage: ERROR!!!")
    return passage_list


default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)

render = web.template.render('templates/')
urls = (
    '/', 'index',
    '/mrc', 'mrc',
)

mrc_form = form.Form(
    form.Textarea("query", description="Query"),
    form.Textarea("passage", description="Passage"),
    form.Button("submit", type="submit", description="Submit"),
)

class index:
    def GET(self):
        raise web.seeother('/mrc')

class mrc:
    def GET(self):
        f = mrc_form()
        get_value = web.input()
        highlight_passage, highlight_answer = "", ""
        passage_list = []
        return render.mrc(f, highlight_passage, highlight_answer, passage_list)

    def POST(self):
        f = mrc_form()
        post_value = web.input(query=None, passage=None)
        if hasattr(post_value, 'random'):
            f['query'].value, f['passage'].value, highlight_passage, highlight_answer = get_an_example()
            passage_list = get_passage(question=f['query'].value)[:5]
            return render.mrc(f, highlight_passage, highlight_answer, passage_list)
        query, passage = post_value.query, post_value.passage
        passage_list = get_passage(question=query)[:5]
        if passage == "":
            passage = passage_list[0][0] if len(passage_list) > 0 else ""
        if query and passage:
            _, _, highlight_passage, highlight_answer = any_input(query, passage)
        else:
            highlight_passage, highlight_answer = "", ""
        f['query'].value, f['passage'].value = query, passage
        return render.mrc(f, highlight_passage, highlight_answer, passage_list)


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()

"""



query_list = [
    "why do i have swollen knee",
    "why do i have itchy eyelids",
    "can pregnant women eat raw seafood",
    "can pregnant women drink red wine",
    "can lemon juice help with a blood pressure",
    "why does my baby snore"
]

default_query = "why do i sneeze after eating"
class demo:
    def GET(self):
        f = demo_form()
        get_value = web.input(p=None)
        p = get_value.p
        if p == None:
            f['query'].value = default_query
        else:
            p = int(p)
            if p < 38:
                f['query'].value = query_list[p]
            else:
                f['query'].value = default_query
        print("[Query]", f['query'].value)
        cards_string = get_cards_from_bing(f['query'].value)
        readout_list = get_readout_from_bing(f['query'].value)
        tts_file_path = get_tts(f['query'].value, readout_list)
        status = False
        return render.demo(f, status, cards_string, readout_list, tts_file_path, query_list)

    def POST(self):
        f = demo_form()
        post_value = web.input(query=None)
        if hasattr(post_value, 'random'):
            post_value.query = query_list[random.randint(0, len(query_list)-1)]
        else:
            if post_value.query == "":
                post_value.query = default_query
        f['query'].value = post_value.query
        print("[Query]", f['query'].value)
        cards_string = get_cards_from_bing(post_value.query)
        readout_list = get_readout_from_bing(post_value.query)
        tts_file_path = get_tts(post_value.query, readout_list)
        status = True
        return render.demo(f, status, cards_string, readout_list, tts_file_path, query_list)


"""


