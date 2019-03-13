import requests
import json

classify_request_dict = {
    "subject": 'Разобраться куда делось место на avi\serv-1c',
    "description": 'Папка Users занимает 43Гб, а юзеры по отдельности только около 2.\r\nРазобраться куда делось остальное.\r\nНа Валеру, или Никиту.\r\n \r\ncid:image001.png@01CEE637.D9119140\r\nЛапин Артём Николаевич\r\nООО <АВИ Консалт>\r\nТел: +7(495)644-4235\r\nE-mail: <mailto:a.lapin@aviconsult.ru> a.lapin@aviconsult.ru\r\nСайт: <http://www.aviconsult.ru/> www.aviconsult.ru\r\n125040, г. Москва, 3-я улица Ямского поля, д.2 кор.3, оф 204	'
}

def classify(classify_request_dict):
    url = "http://192.168.112.1:5012/classify"
    auth_data = ("avi_user", "gmid812375012")
    json_string = json.dumps(classify_request_dict)
    resp = requests.get(url, json=json_string, auth=auth_data)
    return resp

update_dict = {
    'ticket_uid_b64': 'gF4ADCm+Ce4R4oWdR/IpBA==\n',
     'subject': 'aweqqwe',
     'description': 'Внедрить логотип АВИ в Тимвьюер.\r\nВыпустить распоряжение на использование конкретного дистрибутива.\r\nРаспоряжение и правка регламентов на установку Тим.\r\n\r\n \r\n <http://www.aviconsult.ru/> cid:image001.png@01D21264.CDE0B150\r\nЛапин Артём Николаевич\r\nТехнический директор\r\nООО <АВИ Консалт>\r\nТел:  +7(495)644-4235, доб. 221\r\nТел:  +7(495)236-7320, доб. 221\r\nМоб: +7(906)764-3174\r\nE-mail:  <mailto:a.lapin@aviconsult.ru> a.lapin@aviconsult.ru\r\nСайт:  <http://www.aviconsult.ru/> www.aviconsult.ru\r\n125040,  г. Москва, 3-я улица Ямского поля, д.2 кор.3, оф 204',
     'ticket_num': '0000038324',
     'group_name': 'Менеджеры техподдержки',
     'service_name': 'Вспомогательные услуги (безвозмездные)',
     'type_name': 'Запрос на обслуживание',
     'service_part_name': 'Ведение внутренних ИС',
     'priority_name': '2_Стандартный',
     'group_uid_b64': 'u7cADCmJV3wR4Wqtj1RkCw==\n',
     'service_uid_b64': 'moMADCmJV3wR4ZuK3Aaj3A==\n',
     'type_uid_b64': 'h7EzVycVuktKDuRsVkTMbg==\n',
     'service_part_uid_b64': 'jc0AFV1xZAIR5WjgGZrerg==\n',
     'priority_uid_b64': 'u7cADCmJV3wR4WnMb3gohQ==\n',
     'source_name': 'avi'
}

insert_dict = {
    'ticket_uid_b64': 'gF4ADCm+Ce4R4oWdR/IpBA==\n',
     'subject': 'aweqqwe',
     'description': 'Внедрить логотип АВИ в Тимвьюер.\r\nВыпустить распоряжение на использование конкретного дистрибутива.\r\nРаспоряжение и правка регламентов на установку Тим.\r\n\r\n \r\n <http://www.aviconsult.ru/> cid:image001.png@01D21264.CDE0B150\r\nЛапин Артём Николаевич\r\nТехнический директор\r\nООО <АВИ Консалт>\r\nТел:  +7(495)644-4235, доб. 221\r\nТел:  +7(495)236-7320, доб. 221\r\nМоб: +7(906)764-3174\r\nE-mail:  <mailto:a.lapin@aviconsult.ru> a.lapin@aviconsult.ru\r\nСайт:  <http://www.aviconsult.ru/> www.aviconsult.ru\r\n125040,  г. Москва, 3-я улица Ямского поля, д.2 кор.3, оф 204',
     'ticket_num': '1',
     'group_name': 'Менеджеры техподдержки',
     'service_name': 'Вспомогательные услуги (безвозмездные)',
     'type_name': 'Запрос на обслуживание',
     'service_part_name': 'Ведение внутренних ИС',
     'priority_name': '2_Стандартный',
     'group_uid_b64': 'u7cADCmJV3wR4Wqtj1RkCw==\n',
     'service_uid_b64': 'moMADCmJV3wR4ZuK3Aaj3A==\n',
     'type_uid_b64': 'h7EzVycVuktKDuRsVkTMbg==\n',
     'service_part_uid_b64': 'jc0AFV1xZAIR5WjgGZrerg==\n',
     'priority_uid_b64': 'u7cADCmJV3wR4WnMb3gohQ==\n',
     'source_name': 'avi'
}

def update_entry(update_dict):
    url = "http://192.168.112.1:5012/update"
    auth_data = ("avi_user", "gmid812375012")
    json_string = json.dumps(update_dict)
    resp = requests.post(url, json=json_string, auth=auth_data)
    return resp


if __name__ == "__main__":
    # resp = transcribe_local_file(path_to_test_wav)
    resp = classify(classify_request_dict)
    print(resp.json())
    """
    resp = update_entry(insert_dict)
    print(resp.json())
    """