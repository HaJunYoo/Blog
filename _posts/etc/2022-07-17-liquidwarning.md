---
toc: true
layout: post
comments: true
title: Jekyll에서 liquid warning(Liquid Exception) 처리
description: Jekyll fastpages 에서 liquid warning(Liquid Exception) 처리
categories: [etc]
image:
render_with_liquid: false
---

{% raw %}

# Jekyll에서 liquid warning 처리

> `Liquid Exception: Liquid syntax error (line 46): Unknown tag 'url' in markdown(md)`


예제를 작성하거나 할 때에 liquid syntax를 포함시켜야 하는 경우가 있는데.. 그대로 작성하게 되는 경우에 해당 tag가 동작을 해서

```
Liquid Exception: Liquid syntax error (line 47): Tag
```

과 같은 에러가 발생을 하거나 원하지 않는 동작을 하는 경우가 발생한다. 이런 경우에는 raw tag를 이용하면 된다.


![그림1]({{site.baseurl}}/images/etc/rawtag2.png)

Jekyll에서 liquid warning 처리하는 방법은 다음과 같습니다.

Jekyll 에서 사용되는 liquid가 `{{`와 `}}` , `{% %}`를 escape 문자로 사용합니다. 문서에 {{, }} 가 들어 있는 경우 jekyll engine이 경고 메시지를 출력하고, {{ … }} 사이에 있는 내용은 무시됩니다.

문서에는 `x-success={{drafts://}}` 라는 문장이 들어 있습니다.

`Liquid Exception: Liquid syntax error (line 46): Unknown tag 'url' in markdown(md)`

해당 내용을 liquid parsing을 하지 않기 위해서는 문장 앞뒤로 다음과 같은 tag를 추가해 주면 warning과 출력 문제를 해결할 수 있습니다.

---

### 결론

아예 문서 시작 전에 

`raw tag` 를 사용하고

문서 끝에 `raw tag` 를 사용하자

![]({{site.baseurl}}/images/etc/rawtag1.png)

{% endraw %}