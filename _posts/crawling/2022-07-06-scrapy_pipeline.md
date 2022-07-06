---
toc: true
layout: post
comments: false
title: Scrapy pipeline을 통한 필터링 저장
description: Scrapy pipeline을 통해 원하는 데이터만 크롤링
categories: [algorithm]
image:
---

# Scrapy pipeline

> Scrapy 프로젝트를 생성하면 `pipelines.py`가 제공되는데 해당 파일의 역활을 이해해보자.

- pipelines.py 역할은 아이템 데이터 후처리하는 것에 포커스가 맞춰져 있다. 
  - 일부 아이템은 저장
  - 중복되는 아이템을 저장
  - 데이터베이스등에 저장
  - 특별한 포멧으로 아이템을 저장

  <br>
  
- pipelines.py & spider

  - 간단한 크롤링의 경우, 
    - 해당 spider의 parse 함수에서 pipelines.py 역할을 처리할 수 있다.
      - 원하는 데이터만 `yield`를 호출하면 됩니다.
    - 다만, 복잡하고 방대한 크롤링의 경우, 별도 파일에 작성할 수 있도록 되어 있다.

### settings.py 주석 처리 해제

**settings.py** <br>
아래의 코드 주석 처리를 풀어주자.
```python
ITEM_PIPELINES = {
    'mycrawler.pipelines.MycrawlerPipeline': 300,
} # 프로젝트가 만들어 지면 알아서 생성이 된다
# 위의 300은 우선 순위 번호로, 1000 이하의 양의 정수 중 임의로 숫자를 부여하면 됩니다, 
# 여러 클래스가 존재할 때, 숫자가 낮으면 낮을 수록 먼저 실행이 됩니다. 
```
cd 명령어를 통해 mycrawler(크롤링) 폴더로 가서 아래의 명령어를 수행 <br>
`scrapy crawl test_web -o test_web.json -t json`  를 실행해보자. 


```python
2022-07-07 01:47:01 [scrapy.middleware] INFO: Enabled item pipelines:
['mycrawler.pipelines.MycrawlerPipeline']
```
위와 같은 라인을 발견할 수 있다. <br>
즉, 위의 settings.py 주석 처리를 해제하면 
크롤링을 돌렸을 때, 파이프라인이 지원이 되는지 터미널에서 확인이 가능하다. 


### pipelines.py 의 역활?

`from scrapy.exceptions import DropItem` 을 import 해주자.
아래의 코드 양식은 저장해져 있다.

1. 각 아이템 생성 시, pipeline.py 에 있는 `process_item`을 호출
2. 필요한 아이템만 return해준 후, 필터링할 아이템은 `raise DropItem('메세지')`을 통해 처리하지 않도록 해줌
3. 크롤링 -> 후처리 -> 원하는 아이템만 필터하여 저장이 가능

```python
from scrapy.exceptions import DropItem

class MycrawlerPipeline(object):
    def process_item(self, item, spider):
        if item['product_type'] == '제외하고 싶은 아이템 타입':
            raise DropItem('drop item for hanger door')
        else:
            return item
```

다음과 같이 실행 후, 문자열로 지정한 상품은 저장되지 않음을 확인할 수 있습니다. 