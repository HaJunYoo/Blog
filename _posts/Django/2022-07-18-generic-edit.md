---
toc: true
layout: post
comments: true
title: Generic editing view
description: Generic editing view in Django
categories: [Django]
image:
render_with_liquid: false
---

# Generic editing view

## **Generic editing views**

- **폼을 통해 객체를 생성, 수정, 삭제하는 기능을 제공하는 뷰**
- FormView, CreateView, UpdateView, DeleteView
    - **FormView**
    - **CreateView** : post_new()를 구현할 수 있다.
    - **UpdateView** : post_edit()를 구현할 수 있다.
    - **DeleteView** : post_delete()를 구현할 수 있다.

**FormView**

- TemplateResponseMixin, BaseFormView

**CreateView**

- SingleObjectTemplateResponseMixin
- BaseCreateView ← ModelFormMixin, ProcessFormView

**UpdateView**

- SingleObjectTemplateResponseMixin
- BaseUpdateView ← ModelFormMixin, ProcessFormView

**DeleteView**

- SingleObjectTemplateResponseMixin
- BaseDeleteView ← DeletionMixin, BaseDetailView

# FBV

### Create

**post_new()**

```python
@login_required
def post_new(request):
    if request.method == 'POST':
				# forms.py의 PostForm
        form = PostForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user # 현재 로그인한 유저 Instance
            post.save()
            messages.success(request, '포스팅을 저장했습니다.')
						# post의 detail.html로 이동함.
						# redirect는 view 함수 내에서 특정 url로 이동하고자 할 때 사용
						# model에 지정된 get_absolute_url로 이동
						return redirect(post)
    else:
        form = PostForm()

    return render(request, 'instagram/post_form.html',{
        'form' : form,
				 # 새 글을 작성할 당시 post는 Null 값일수도 있기에 None을 리턴한다.
		    'post' : None,
    })
```

### Update

**post_edit()**

```python
@login_required
def post_edit(request, pk):
		# get_object_or_404는 model을 첫번째 인자로 받고, 몇개의 키워드 인수를 get() 함수에 넘김
    post = get_object_or_404(Post, pk=pk)

		# 이렇게 반복되는 코드가 있다면 장식자로 만드는 것도 좋은 방법
		# 작성자만 수정할 수 있도록
		if post.author != request.user:
        messages.error(request, '작성자만 수정 가능합니다!')
        return redirect(post)

    if request.method == 'POST':
        form = PostForm(request.POST, request.FILES, instance=post)
        if form.is_valid():
            post = form.save()
            messages.success(request, '포스팅을 수정했습니다.')
						
						# post의 detail.html로 이동함.
						return redirect(post)
    else:
        form = PostForm(instance=post)

    return render(request, 'instagram/post_form.html',{
										       'form' : form,
														# 수정한다는 것은 이미 원글이 있다는 뜻이기에 post를 리턴한다.
														'post' : post,
    })
```

### Delete

**post_delete()**

```python
@login_required
def post_delete(request, pk):
		# get_object_or_404는 model을 첫번째 인자로 받고, 몇개의 키워드 인수를 get() 함수에 넘김
    post = get_object_or_404(Post, pk=pk)

		# 유저가 삭제 확인을 했을 때는 
		if request.method == 'POST':
        post.delete()
        messages.success(request, '성공적으로 삭제하였습니다.')
        return redirect('instagram:post_list')
    
	return render(request, 'instagram/post_confirm_delete.html', {
        'post' : post,

    })
```

---

# CBV

### FormView

- 폼이 주어지면 해당 폼을 출력
- **TemplateResponseMixin**과 **BaseFormView**를 상속받는다

**Example myapp/forms.py**:

```python
from django import forms

class ContactForm(forms.Form):
    name = forms.CharField()
    message = forms.CharField(widget=forms.Textarea)

    def send_email(self):
		# send email using the self.cleaned_data dictionary
			pass
```

**Example myapp/views.py**:

```python
from myapp.forms import ContactForm
from django.views.generic.edit import FormView

class ContactFormView(FormView):
    template_name = 'contact.html'
    form_class = ContactForm
    success_url = '/thanks/'

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        form.send_email()
        return super().form_valid(form)
        
    def get_success_url(self):
	    	# 주의 : Post 모델에 get_absolute_url() 멤버함수 구현 필요
        return resolve_url(self.object)
        
        # return self.post.get_absolute_url() 대안 1
        # return reverse('blog:post_detail', args=[self.post.id]) 대안 2
        
 post_new = PostCreateView.as_view()
```

**Example myapp/contact.html**:

```python
<form method="post"> {% csrf_token %}
    {{ form.as_p }}
    <input type="submit" value="Send message">
</form>
```

---

## Create

### CreateView

- 객체를 생성하는 폼 출력
- `CreateView`에서 템플릿명 디폴트는 `{{model_name}}_form.html`로 지정된다.

**instagram/views.py**

```python
from django.views.generic import CreateView
from myapp.models import Post

# 클래스형 뷰에서 Create폼 처리하기
class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    form_class = PostForm

    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.author = self.request.user # 현재 로그인한 유저 Instance
        messages.success(self.request, '포스팅을 저장했습니다.')
        return super().form_valid(form)

post_new = PostCreateView.as_view()
```

**Example myapp/author_form.html**:

```python
<form method="post">
	{% csrf_token %}
    {{ form.as_p }}
  <input type="submit" value="Save">
</form>
```

---

### UpdateView

- 기존 객체를 수정하는 폼을 출력
- UpdateView에서 템플릿명 디폴트는 `{{model_name}}_form.html` 로 지정된다.

**instagram/views.py**

```python
from django.views.generic.edit import UpdateView
from myapp.models import Post

# 클래스형 뷰에서 Update폼 처리하기
class PostUpdateView(LoginRequiredMixin, UpdateView):
    model = Post
    form_class = PostForm

    def form_valid(self, form):
        messages.success(self.request, '포스팅을 저장했습니다.')
        return super(PostUpdateView, self).form_valid(form)

post_edit = PostUpdateView.as_view()
```

**Example myapp/author_update_form.html**:

```xml
<form method="post">{% csrf_token %}
    {{ form.as_p }}
    <input type="submit" value="Update">
</form>
```

---

### DeleteView

- 기존 객체를 삭제하는 폼을 출력
- DeleteView에서 템플릿명 디폴트는 `{{model_name}}_confirm_delete.html`로 지정된다.

**Example myapp/views.py**:

```python
from django.urls import reverse_lazy
from django.views.generic.edit import DeleteView
from myapp.models import post

# 클래스형 뷰에서 Delete 처리하기
class PostDeleteView(LoginRequiredMixin, DeleteView):
    model = Post
    success_url = reverse_lazy('instagram:post_list')
		# reverse_lazy는 실제 값이 사용될 때 reverse를 해줌으로 오류 예방
    template_name = 'instagram/post_confirm_delete.html'

    def delete(self, request, *args, **kwargs):
        response = super(PostDeleteView, self).delete(request, *args, **kwargs)
        messages.success(request, '포스팅을 삭제했습니다.')
        return response

post_delete = PostDeleteView.as_view()
```

- **success_url**
    - 지정된 개체가 삭제되었을 때 리디렉션할 url
    - `/parent/{parent_id}/` 처럼 사전 문자열 형식을 포함할 수 있다

```python
class PostDeleteView(LoginRequiredMixin, DeleteView):
    model = Post
    
		# reverse_lazy를 쓰지 않을 경우 아래와 같이 사용
    def get_success_url(self):
    	return reverse('instagram:post_list')
```

**post_delete()** 함수의 **return redirect('instagram:post_list')**와 같은 역할을 한다.

**Example myapp/author_confirm_delete.html**:

```python
<form method="post">{% csrf_token %}
    <p>Are you sure you want to delete "{{ object }}"?</p>
    {{ form }}
    <input type="submit" value="Confirm">
</form>
```

--------
##### Reference

- [파이썬/장고 웹서비스 개발 완벽 가이드 with 리액트](https://www.inflearn.com/course/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%9E%A5%EA%B3%A0-%EC%9B%B9%EC%84%9C%EB%B9%84%EC%8A%A4/dashboard)

- [https://devdongbaek.tistory.com/91?category=1037491](https://devdongbaek.tistory.com/91?category=1037491)