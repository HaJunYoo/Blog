---
toc: true
layout: post
comments: false
title: 부분집합 구하기(DFS)
description: 깊이 우선 탐색을 이용한 부분집합 구하기
categories: [algorithm]
image: images/algorithm/부분집합구하기.jpeg
---

# 부분 집합 구하기(DFS)

자연수 N이 주어지면 1부터 N까지의 원소를 갖는 집합의 부분집합을 모두 출력하는 프로그램을 작성하세요

- 입력 설명
    - 첫번째 줄에 자연수 N(1≤ N ≤ 10)이 주어집니다.
- 출력 설명
    - 첫 번째 줄부터 각 줄에 하나씩 아래와 같이 출력한다. 출력순서는 깊이 우선 탐색 전위 순회 방식으로 출력합니다. 단 공집합은 출력하지 않습니다.

- 입력 예제
    - 3
    
- 출력 예제
    - 1 2 3
    - 1 2
    - 1 3
    - 1
    - 2 3
    - 2
    - 3

---

`DFS`를 잘하려면 다음과 같은 상태트리를 잘 사용하면 된다. 

D(1) → D(2) 를 사용하냐 or 사용하지 않나… 로 나눈다.

![경우의수]({{site.baseurl}}/images/algorithm/부분집합구하기.jpeg)

모든 노드를 왼쪽 자식, 오른쪽 자식이 아닌 번호로 생각한다 

(v, v*2 , v*2+1) 이 아닌 `(v, v+1, v+1)`

빈 리스트를 만들어 해당 인덱스를 사용하면 1로 표시해주고 아닐 시 0으로 표시해준다. 

깊이우선 탐색 방식으로 

v가 4가 되면 (인덱스 초과), 인덱스가 1인 리스트의 원소만 출력을 해준다. 

| ch_index | 1 | 2 | 3 |
| --- | --- | --- | --- |
| ch | 1 | 1 | 1 |

⇒ print ⇒ 1 2 3

다시 백트래킹해서 D(3)으로 돌아와 인덱스 3을 사용하지 않는다로 표시해준다.

그리고 v+1 해줘서 v가 4가 되어 인덱스 초과 시 해당 리스트를 출력해준다.

| ch_index | 1 | 2 | 3 |
| --- | --- | --- | --- |
| ch | 1 | 1 | 0 |

⇒ print ⇒ 1 2

D(1, 2) = 1 일 때, D(3)의 경우의 수가 다 해결되었다면 백트래킹으로 D(2)로 돌아와 아래의 리스트를 해결하고

해당 방식을 반복해준다. 

| ch_index | 1 | 2 | 3 |
| --- | --- | --- | --- |
| ch | 1 | 0 | 1, 0 |

```python
n = int(input())

ch = [0]*(n+1)

def DFS(v):
    if v == n+1 :
        for idx, elem in enumerate(ch) : # 원소가 1인 경우만 인덱스 출력
            if elem != 0 : 
                print(idx, end = ' ')
        print()

    else :
        ch[v] = 1 # 사용한다
        DFS(v+1)
        ch[v] = 0 # 시용하지 않는다
        DFS(v+1) 

DFS(1) # 1부터 스택을 쌓아서 입력을 넘지 않을 때까지 깊이 우선 탐색을 진행한다. 
```
