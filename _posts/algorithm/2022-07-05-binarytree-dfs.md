---
toc: true
layout: post
comments: false
title: 이진트리 순회(DFS)
description: 깊이 우선 탐색을 이용한 이진 트리 순회
categories: [algorithm]
image: images/algorithm/이진트리순회.png
---
# 이진트리 순회 (DFS)

아래 그림과 같은 `이진트리`를 전위순회와 후위순회를 연습하자.

![그림 1]({{site.baseurl}}/images/algorithm/이진트리순회.png)

트리는 루트를 기반으로 왼쪽 자식부터 오른쪽 자식으로 탐색을 한다. 

해당 탐색을 `깊이 우선적`으로 탐색한다.

`전위순회`는 

( 루트(부모) - 왼쪽 자식 - 오른쪽 자식) ⇒ 재귀적으로 반복

전위순회 출력 : 

(1-(2-4-5)-(3-6-7))

`중위 순회`는

(왼쪽 자식 - 부모 - 오른쪽 자식) ⇒ 재귀적으로 반복한다

중위순회 출력 : 

((4- 2- 5) - 1 - (6- 3- 7))

`후위 순회`는

(왼쪽 자식 - 오른쪽 자식 - 부모 ) ⇒ 재귀적으로 반복한다

후위순회 출력 : 

((4-5-2) - (6-7-3) - 1)

![그림 2]({{site.baseurl}}/images/algorithm/이진트리순회1.png)

`부모(n)`를 기준으로 

왼쪽 자식은 `2n`

오른쪽 자식은 `2n+1`

![그림 3]({{site.baseurl}}/images/algorithm/이진트리순회2.png)

**vertex** ⇒ **node**

```python
def DFS(v) : # v는 부모 vertex(node)
	if v > 7 :
		return # 7보다 클 경우 함수 종료

	else :
		DFS(v*2) # 왼쪽 자식 노드 호출
		DFS(v*2+1) # 오른쪽 자식 노드 호출 

DFS(1) # 1번 노드부터 호출 시작
```

보통 이진 트리의 node들은 리스트의 형식으로 저장해준다.  

`tree = [1,2,3,4,5,6,7]`

하지만 아래 코드에서는 위의 리스트를 사용하지 않을 것이다.

### 전위 순회 코드

```python
def DFS(v) : # v는 부모 vertex(node)
	if v > 7 :
		return # node가 7보다 클 경우 함수 종료

	else :
		print(v, end = " ") # 부모 출력 
		DFS(v*2) # 왼쪽 자식 노드 호출
		DFS(v*2+1) # 오른쪽 자식 노드 호출 

DFS(1) # 1번 노드부터 호출 시작
# 1 2 4 5 3 6 7
```

### 중위 순회 코드

```python
def DFS(v) : # v는 부모 vertex(node)
	if v > 7 :
		return # node가 7보다 클 경우 함수 종료

	else :
		DFS(v*2) # 왼쪽 자식 노드 호출
		print(v, end = " ") # 부모 출력
		DFS(v*2+1) # 오른쪽 자식 노드 호출 

DFS(1) # 1번 노드부터 호출 시작
# 4 2 5 1 6 3 7
```

### 후위 순회 코드

```python
def DFS(v) : # v는 부모 vertex(node)
	if v > 7 :
		return # node가 7보다 클 경우 함수 종료

	else :
		DFS(v*2) # 왼쪽 자식 노드 호출
		DFS(v*2+1) # 오른쪽 자식 노드 호출 
		print(v, end = " ") # 부모 출력

DFS(1) # 1번 노드부터 호출 시작
# 4 5 2 6 7 3 1
```

후위 순회를 사용하는 DFS 중 대표적인 문제가 `병합정렬`

순회 방식에서 사용 빈도는  **전위 >> 후위 > 중위**
