from bs4 import BeautifulSoup
import requests
import csv
import matplotlib.pyplot as plt
import re

count_lists=list()
fields = ['Div_Count','Span_Count','Link_Count','Para_Count',
            'List_Count','Img_Count','Btn_Count','Script_Count','Iframe_Count','Table_Count','Article_Count','Word_Count','Link','Article']  #change here

filename = "samsung_prism_art.csv"  
mydict_list =[]
            

def scrapUrl(s):
  articleAmar_url=s #place new article links here
  amar_re=requests.get(articleAmar_url)

  amarContent=amar_re.content
  amarSoup=BeautifulSoup(amarContent,'html.parser')
  #print(amarSoup.prettify)

  #find div tag of article
  amarTitle=amarSoup.title
  print(amarTitle.text)


  special_char=re.compile('[@_!$%^&*()<>?/\}{~:]#.')#new section for word count

  amar_div=amarSoup.find_all('div')
  div_count=0
  div_words=0
  dw=""
  for divs in amar_div:
    if (divs!=""):
      div_count=div_count+1
      dw=divs.text.split()
      if((len(dw)>=10) and (special_char.search(divs.text) == None)):
        div_words+=len(dw)
     ## print("Div Text:{}".format(divs.text))

  count_lists.append(div_count)

  print("div word:",div_words)
  print("div count:",count_lists)

  amar_span=amarSoup.findAll('span')
  span_word=0
  sw=""
  span_count=0
  for spans in amar_span:
    if(spans !=""):
      span_count=span_count+1
      sw=spans.text.split()
      if ((len(sw)>=10)and (special_char.search(spans.text)== None)):
        span_word+=len(sw)
      ##print("spans_data: {}".format(spans.text))

  print("span word",span_word)
  print("span_count",span_count)
  count_lists.append(span_count)

  #find content of anchor tag and inner text 
  amar_anchor=amarSoup.findAll('a')
  link_count=0
  text_ref=[]
  for anchors in amar_anchor:
    if(anchors!='#'):
      link_count=link_count+1
      text_anc=anchors.get_text()
      text_ref=anchors.get('href')
      #print(type(text_ref))
     ## print("Links: {}".format(text_ref))

  print("Link count",link_count)
  count_lists.append(link_count)
  #find paragraph in article 
  amar_para=amarSoup.findAll('p')
  para_word=0
  pw=""
  para_count=0
  for para in amar_para:
    if(para.text!=""):
      para_count=para_count+1
      pw=para.text.split()
      if ((len(pw)>=10)and (special_char.search(para.text)== None)):
        para_word+=len(pw)
     ## print(para.text)
  print("Para Word",para_word)
  print("Para_count",para_count)
  count_lists.append(para_count)

  amar_li=amarSoup.findAll('li')
  list_word=0
  lw=""
  list_count=0
  for lists in amar_li:
    if(lists!=""):
      list_count=list_count+1
      lw=lists.text.split()
      if((len(lw)>=10)and (special_char.search(lists.text)==None)):
        list_word+=len(lw)
      ##print("List_data: {}".format(lists.text))
  print("List Word",list_word)
  print("list_count",list_count)
  count_lists.append(list_count)

  #image count 
  img_count=0
  for img in amarSoup.find_all('img'):
    img_count+=1
    ##print(img)
  print("Image Count",img_count)
  count_lists.append(img_count)

  #Button Count
  btn_count=0
  for btn in amarSoup.find_all('button'):
    btn_count+=1
    ##print(img)
    #print(btn.text)
  print("Button Count",btn_count)

  #Script Count
  amar_script=amarSoup.findAll('script')#working
  script_count=0
  for scripts in amar_script:
    script_count=script_count+1
  print('Script Count',script_count)

  #Iframe Count
  amar_iframe=amarSoup.findAll('iframe')
  frame_count=0
  for frame in amar_iframe:
    frame_count+=1
  print("iframe Count",frame_count)

  #table Count
  amar_table=amarSoup.findAll('table')
  table_count=0
  for tables in amar_table:
    table_count+=1
  print('Table Count',table_count)

  #Article Count
  amar_article=amarSoup.findAll('article')
  art_word=0
  aw=""
  article_count=0
  for articles in amar_article:
    article_count+=1
    aw=articles.text.split()
    if((len(aw)>=10)and (special_char.search(articles.text)==None)):
      art_word+=len(aw)

  print("Article wors",art_word)
  print("Article Count",article_count)


  sum_words=div_words+span_word+para_word+art_word
  print("Total Word count",sum_words)
  count_lists
  # tags=["div","span","link","para","list","img"]

  # fig=plt.figure(figsize = (10, 5))
  # plt.bar(tags, count_lists, color ='red',width = 0.4)
  # plt.xlabel("Tags Name")
  # plt.ylabel("Tags Count")
  # plt.title(line)
  # plt.show()
  count_lists.clear()
  
  case = {'Div_Count':div_count,'Span_Count':span_count,
              'Link_Count':link_count,'Para_Count':para_count,
              'List_Count':list_count,'Img_Count':img_count,
              'Btn_Count':btn_count,'Script_Count':script_count,
              'Iframe_Count':frame_count,'Table_Count':table_count,
          'Article_Count':article_count,'Word_Count':sum_words,'Link':s,'Article':"YES"}#changed here

  mydict_list.append(case)




file1 = open('/content/testingArt.txt', 'r')  #Put your text file here
count = 0

while True:
    count += 1
 
    # Get next line from file
    line = file1.readline()
 
    # if line is empty
    # end of file is reached
    if not line:
        break
    print("Line{}: {}".format(count, line.strip()))
    scrapUrl(line.strip())

 
    
 
file1.close()

with open(filename,'w',newline='')as csvfile:
      writer = csv.DictWriter(csvfile,fieldnames = fields)
      writer.writeheader() 
      writer.writerows(mydict_list)  




# baseUrl=input('Enter URL')

# scrapUrl(baseUrl)

# count_lists
# tags=["div","span","link","para","list","img"]

# fig=plt.figure(figsize = (10, 5))
# plt.bar(tags, count_lists, color ='red',width = 0.4)
# plt.xlabel("Tags Name")
# plt.ylabel("Tags Count")
# plt.title(baseUrl)
# plt.show()
