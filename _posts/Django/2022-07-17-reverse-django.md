---
toc: true
layout: post
comments: true
title: URL Reverse를 통해 유연하게 URL 생성
description: URL Reverse를 통해 유연하게 URL 생성하기 - Django
categories: [Django]
image:
render_with_liquid: false
---


# URL Dispatcher

장고는 `urls.py`를 통해 **각 뷰에 대한 URL이 변경되는 유연한 URL 시스템**을 갖는다.

뷰에 대한 이름은 path에 name 파라미터를 지정하면 된다.

`app_name`은 앱 내의 네임스페이스 역할을 한다. post_list가 다른 앱에 존재하더라도 `app_name`을 통해 구분할 수 있다.

URL Reverse를 수행함으로써 개발자가 URL을 계산할 필요가 없다. 

추후에 path의 url이 변경되더라도 `URL Reverse가 변경된 url을 추적`하기 때문에 누락될 일이 없는 유연한 대응이 가능하다.

## urls.py 정의

```python
# project/urls.py
urlpatterns = [
    # ...
    path('instagram/', include('instagram.urls')),
]
```

```python
# instagram/urls.py
app_name = 'instagram'  # app_name 설정. 네임스페이스 역할

urlpatterns = [
    path('', views.post_list, name='post_list'),  # name 파라미터 입력. 뷰의 이름.
    path('<int:pk>/', views.post_detail, name='post_detail'),
]
```

## URL Reverse 수행하기

instagram앱의 urls.py에서 정의한 **url**을 활용하는 예시

**템플릿 예시** 및 **reverse**, **resolve_url**, **redirect** 등의 함수를 이용해서 서버코드에서도 활용이 가능하다.

url 내용을 변경하더라도 아래 예시들과 같이 URL Reverse를 사용한 코드는 변경할 필요가 없게된다.

### URL Reverse를 수행하는 4가지 함수

- `url 템플릿태그`
내부적으로 reverse 함수를 사용
    
    `{% url 인자 %}`
    
    ```python
    <a href="instagram/{{post.pk}}">포스트 리스트</a>  
    # (이런 하드코딩은 나중에 URL을 변경 시 함께 변경되어야 한다.) 
    
    <a href="{% url 'instagram:post_list' %}">포스트 리스트</a>
    
    ----------------------------------------------------------
    
    # instagram앱에서 post_detail에 대한 pk=100일 때의 URL을 가져온다. 
    
    # (`instagram/100`)파라미터 이름(pk)는 생략이 가능하다. 
    
    <a href="{% url 'instagram:post_detail' 100 %}">포스트 리스트</a>
    
    <a href="{% url 'instagram:post_detail' pk=100 %}">포스트 리스트</a>
    
    <a href="{% url 'instagram:post_detail' post.pk %}">포스트 리스트</a>
    ```
    
- `reverse 함수`
    
    반환값 : URL 문자열
    
    args 또는 kwargs라는 파라미터 이름을 지정해서 정해진 타입으로 입력해야 한다. 
    
    (`args:` **리스트** - 여러 개일 경우 파라미터 순서대로 지정, `kwargs:` **딕셔너리** - 파라미터 이름 지정)
    
    매칭 URL이 없으면 NoReverseMatch 예외 발생
    
    ```python
    from django.urls import reverse
    
    reverse('instagram:post_detail', args=[100])  # '/instagram/100/' 반환
    reverse('instagram:post_detail', kwargs={'pk': 100})  # '/instagram/100/' 반환
    ```
    
- `resolve_url 함수`
    
    반환값 : URL 문자열
    
    reverse를 래핑해서 편리하게 사용할 수 있도록 만든 함수. 
    
    동적으로 여러개의 파라미터를 입력하거나 파라미터 이름을 지정해서 입력한다.
    
    매핑 URL이 없으면 "인자 문자열"을 그대로 리턴
    내부적으로 **reverse 함수**를 사용
    
    ```python
    from django.shortcuts import resolve_url
    
    resolve_url('instagram:post_detail', 100)  # '/instagram/100/' 반환
    resolve_url('instagram:post_detail', pk=100)  # '/instagram/100/' 반환
    resolve_url('/instagram/100/')  # '/instagram/100/' 반환
    ```
    
- `redirect 함수`
    
    반환값 : HttpResponse(301 or 302) 인스턴스를 반환
    뷰에서 특정 로직에 대한 리다이렉트 응답이 필요할 경우 사용가능. 
    
    매칭 URL이 없으면 "인자 문자열"을 그대로 URL로 사용
    내부적으로 **resolve_url 함수**를 사용
    
    ```python
    from django.shortcuts import redirect
    
    redirect('instagram:post_detail', 100)  # /instagram/100/에 대한 HttpResponse 반환
    redirect('instagram:post_detail', pk=100)  # /instagram/100/에 대한 HttpResponse 반환
    redirect('/instagram/100/')  # /instagram/100/에 대한 HttpResponse 반환
    ```
    

## 모델 객체에 대한 detail 주소 계산을 위한 코드 간소화

인자로 `모델 객체(post)`를 넘겨준다

다음 코드를 매번 입력하는 것도 좋지만

```python
resolve_url('instagram:post_detail', pk=post.pk)
redirect('instagram:post_detail', pk=post.pk)
{% url 'instagram:post_detail' post.pk %}
```

아래와 같이 사용할 수 있다.

```python
resolve_url(post) # 파이썬 상
redirect(post) # 파이썬 상
**{{ post.get_absolute_url }} # 템플릿 문법**
```

## 모델 클래스에  **`get_absolute_url()`**  구현

모델 클래스를 만들면 내부 클래스 메소드로 `get_absolute_url` 메서드를 구현하면 된다.

resolve_url 함수는 가장 먼저 `get_absolute_url()` 함수의 존재 여부를 확인하고 존재한다면 reverse를 수행하지 않고 그 리턴값을 즉시 리턴하게 되어 있다.

redirect 또한 내부적으로는 resolve_url을 사용하기에 get_absolute_url() 구현 여부에 따라 영향을 받는다.

```python
class Post(models.Model):
    # ...
    def get_absolute_url(self):
        return reverse('instagram:post_detail', args=[self.pk])
```

## 기타

- **CreateView**, **UpdateView와** 같은 Generic CBV에서 success_url을 지정할 수 있다.
    
    만약 지정하지 않았을 경우, 해당 모델 인스턴스의 `get_absolute_url()` 주소로 이동 가능한지 여부를 체크하고, 이동이 가능할 경우 이동한다.
    
- 특정 모델에 대한 Detail 뷰를 작성할 경우, Detail view에 대한 url configure 설정을 하자마자, `get_absolute_url()`을 구현할 것을 추천한다. 코드가 간결해 진다.


---

### Reference

[https://velog.io/@joje/URL-Reverse를-통해-유연하게-URL-생성하기](https://velog.io/@joje/URL-Reverse%EB%A5%BC-%ED%86%B5%ED%95%B4-%EC%9C%A0%EC%97%B0%ED%95%98%EA%B2%8C-URL-%EC%83%9D%EC%84%B1%ED%95%98%EA%B8%B0)

****[파이썬/장고 웹서비스 개발 완벽 가이드 with AskCompany](https://velog.io/@joje/series/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9E%A5%EA%B3%A0-%EC%9B%B9%EC%84%9C%EB%B9%84%EC%8A%A4-%EA%B0%9C%EB%B0%9C-%EC%99%84%EB%B2%BD-%EA%B0%80%EC%9D%B4%EB%93%9C-with-AskCompany) - Inflearn**