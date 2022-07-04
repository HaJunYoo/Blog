---
toc: true
layout: post
comments: false
title: 재귀함수와 스택
description: 재귀함수와 스택을 설명
categories: [algorithm]
image: images/algorithm/파도반.png
---
# 재귀함수와 스택

**반복문을 대체제 : 재귀 함수**

DFS라는 함수로 `재귀함수와 스택`을 리뷰해보자 

```python
def DFS1(x) :
	if x>0 :
		DFS1(x-1)
		print(x, end=' ')
# 1 2 3

def DFS2(x) :
	if x>0 :
		print(x, end=' ')
		DFS2(x-1)

# 3 2 1
```

위의 DFS 1,2 는 재귀 호출 위치에 따라서 리턴값이 각 오름차순과 내림차순으로 다르다. 

메모리 영역 스택에 매개변수가 할당이 된다 

DFS(3) → 스택에 `매개변수 x = 3, 지역변수, 복귀주소`가 할당된다.

이렇게 재귀로 인해 쌓인 메모리 묶음을 `스택 프레임`이라고 한다

![]({{site.baseurl}}/images/algorithm/재귀함수와 스택.png)

DFS(0)이 가장 위에 쌓이는 순간 DFS(0)은 x >0 조건을 성립하지 못하기 때문에 종료가 된다 

종료가 되면서 스택 최상단에 있던 DFS(0)은 없어진다. 

지워지면서 복귀주소 ⇒ DFS(1)로 복귀한다. 

```python
def DFS(x) :
	if x>0 :
		DFS(x-1) ----> 각 복귀 주소
		print(x, end=' ')
# 1 2 3
```

복귀 주소로 복귀한 다음 바로 다음 줄인 `print(x, end = ‘ ‘)` 를 수행한다. 

수행하고 종료되고 메모리가 할당 해제되면서 스택에서 사라진다.! 그리고 다시 다음 DFS(2)가 호출된다

그렇게 DFS(3)까지 호출되고 사라짐을 반복한다. 

그렇게 `1 , 2, 3` 이 출력되는 것이다.

그런 후 스택이 텅 비면 이제 다음 코드 라인으로 이동해서 코드를 수행한다.