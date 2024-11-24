import os

import requests
import base64
import urllib
import json
""" Your APPID AK SK """
APP_ID = ''
API_KEY = ''
SECRET_KEY = ''

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

# text entities
def get_text_entities(text,access_token):
    # 截断文本
    if len(text) > 127:
        text = text[:127]

    url = "https://aip.baidubce.com/rpc/2.0/nlp/v1/entity_analysis?access_token=" + access_token

    payload = json.dumps({
        "text": text
    })

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload).json()
    try:
        entities_analysis = response['entity_analysis']
    except:
        print('wrong：',text)
        print(response)
        # return FALSE
        return False,False,False
    entity_names = []
    entity_introduce = []
    for item in entities_analysis:
        if item['status'] == 'LINKED':
            entity_names.append(item['mention'])
            entity_introduce.append((item['desc']))
        elif item['status'] == 'NIL':
            continue
    return entity_names,entity_introduce,response

# image entities
def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

# api
def get_pic_object(img_path,access_token):
    url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general?access_token=" + access_token

    image = get_file_content_as_base64(img_path, True)

    payload='image=' + image + '&baike_num=5'

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload).json()
    try:
        res_entities = response['result']
    except:
        print('wrong：',img_path)
        print(response)
        return False,False,False
    entity_names = []
    entity_introduce = []
    for item in res_entities:
        if item['score'] > 0.1:
            entity_names.append(item['keyword'])
            if len(item['baike_info']) != 0:
                try:
                    entity_introduce.append(item['baike_info']['description'])
                except:
                    print('出错图片：',img_path)
                    print(res_entities)
                    entity_introduce.append(item['keyword'])
            else:
                entity_introduce.append(item['keyword'])
    return entity_names,entity_introduce,response

def get_img_path(images_name,LABLEF,local_path='weibo'):
    imgs = images_name.split('|')
    for img in imgs:
        if img.split('.')[-1] == 'gif':
            continue
        GT_path = "{}/{}/{}".format(local_path, LABLEF, img)
        if os.path.exists(GT_path):
            return GT_path


def get_objects_form_pics(image_paths,LABLEF,access_token,local_path='./data/datasets/weibo'):
    imgs = image_paths.split('|')
    img_entity_names = []
    img_entity_introduce = []
    for img in imgs:
        if img.split('.')[-1] == 'gif':
            continue
        GT_path = "{}/{}/{}".format(local_path, LABLEF, img)
        if os.path.exists(GT_path):
            this_img_entity_names, this_img_entity_introduce,respos = get_pic_object(GT_path, access_token)
            if this_img_entity_names is not False:
                img_entity_names = img_entity_names + this_img_entity_names
                img_entity_introduce = img_entity_introduce + this_img_entity_introduce
    return img_entity_names,img_entity_introduce

