---
toc: true
layout: post
comments: true
title: 뷰 장식자 (View Decorators)
description: 뷰 데코레이터(view decorator) in Django
categories: [Django]
image:
---

## 장식자 (Decorators)

> 어떤 함수를 감싸는 (Wrapping) 함수

1. 함수를 `장식자`가 wrapping 함
2. 외부에서 장식한 함수를 보면 장식자가 먼저 인식됨 
3. 경우에 따라서 내부 로직에 들어가서 장식한 함수를 호출하기도 함

## View Decorator(뷰 데코레이터) in Django

### FBV 에서의 사용

```python
from django.contrib.auth.decorators	import login_required
from django.shortcuts	import render

@login_required
def protected_view1(request):
	return render(request,	'myapp/secret.html')

def protected_view2(request):
	return render(request,	'myapp/secret.html')

protected_view2	=	login_required(protected_view2)
```

위의 두 함수 1,2 는 동일한 기능을 합니다. 

### 몇 가지 장고 기본 Decorators

`django.views.decorators.http`

- require_http_methods (request_method_list) : 뷰가 특정 요청 메소드만 허용하도록 요구
- require_GET : 뷰가 GET 메소드만 허용하도록 요구
- require_POST : 뷰가 POST 메소드만 허용하도록 요구
- require_safe : 뷰가 GET 및 HEAD 메소드만 허용하도록 요구
- 지정 method가 아닐 경우, `HttpResponseNotAllowed` 응답 (상태코드 405)

`django.contrib.auth.decorators`

- user_passes_test : 지정 함수가 False를 반환하면 login_url로 redirect
- login_required : 로그아웃 상황에서 login_url로 redirect
- permission_required : 지정 퍼미션이 없을 때 login_url로 redirect

`django.contrib.admin.views.decorators`

- staff_member_required : staff member가 아닐 경우 login_url로 이동

### CBV에서의 decorator

##### 1.  `as_view`를 이용해 함수를 만든 후 함수를 감싸줌

```python
from django.contrib.auth.decorators import login_required
from django.views.generic import TemplateView

# TemplateView를 상속
class MyTemplateView(TemplateView):
  template_name = 'core/index.html'

index = MyTemplate.as_view()
index = login_required(index)
```

##### 2. `dispatch` 함수 사용
- `dispatch`는 클래스가 새로운 함수를 만들 때 마다 항상 실행되는 함수
- `dispatch`에 새로운 내용을 추가하는 것 아닌데 재정의하기 때문에 가독성 떨어뜨림
- Class의 멤버 함수에는 method_decorator를 활용
    - 인자에도 decorator 내용을 넣는다.

```python
from django.utils.decorators import method_decorator

class MyTemplateView(TemplateView):
  template_name = 'core/index.html'
  
# Class의 멤버 함수에는 method_decorator를 활용
  @method_decorator(login_required)
  def dispatch(self, *args, **kwargs):
    return super().dispatch(*args, **kwargs)
    
 index = MyTmeplateView.as_view()
```

##### 3. **Class에 직접 적용** ← 가장 권장

- **`@method_decorator`에 `name` 지정해 직접 클래스 메소드에 대해 decorator 사용하기**

```python
@method_decorator(login_required, name='dispatch')
class MyTemplateView(TemplateView):
  template_name = 'core/index.html'
  
index = MyTemplateView.as_view()
```

##### PS. 데코레이터를 사용하지 않고 비슷한 기능의 `LoginRequiredMixin` 상속
- 마찬가지로 사용하려는 데코레이터와 비슷한 기능을 가진 클래스를 상속받아 재정의 가능.

```python
class PostListView(LoginRequiredMixin, ListView):
    model = Post
    paginate_by = 100

post_list = PostListView.as_view()
```


--------
##### Reference

[https://velog.io/@chldppwls12/django-view-decorators](https://velog.io/@chldppwls12/django-view-decorators)