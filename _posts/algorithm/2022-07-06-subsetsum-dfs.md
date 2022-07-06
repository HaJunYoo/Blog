---
toc: true
layout: post
comments: false
title: 합이 같은 부분집합(DFS : 아마존 인터뷰)
description: 깊이 우선 탐색을 이용하여 부분집합 합이 같은 것을 알아보기
categories: [algorithm]
image: 
---
# 합이 같은 부분집합(DFS : 아마존 인터뷰)

N개의 원소로 구성된 자연수 집합이 주어지면, 이 집합을 두 개의 부분집합으로 나누었을 때, 

두 부분 집합의 원소의 합이 서로 같은 경우가 존재하면 “YES”를 출력하고, 그렇지 않으면 “NO”를 출력하는 프로그램을 작성하시오.

예를 들어 {1, 3, 5, 6, 7, 10}이 입력되면 {1, 3, 5, 7} = {6, 10} 으로 두 부분 집합의 합이 16으로 같은 경우가 존재하는 것을 알 수 있다.

- 입력 설명
    - 첫번째 줄에 자연수 N(1≤ N ≤ 10) 이 주어집니다.
    - 두번째 줄에 집합의 원소 N개가 주어진다. 각 원소는 중복되지 않는다.
    
- 출력 설명
    - 첫번째 줄에 “YES” 또는 “NO”를 출력한다.
    
- 입력 예제 1
    - 6
    - 1 3 5 6 7 10
    
- 출력 예제 1
    - YES
    

---

**Total**은 입력받은 수들의 총합

`DFS(index, sum)`

`왼쪽 매개변수(index)`는 리스트의 index

`오른쪽 매개변수 sum`은 index까지의 부분 집합들의 합이다. 

함수는 매 리스트 인덱스 분기점마다 해당 리스트의 원소를 `add or not` 을 선택하면서 리스트의 끝까지 탐색한다.

리스트의 끝까지 탐색을 했고 아래의 조건을 만족하였을 때, 

`Sum == Total - sum` 

⇒ YES

⇒ 프로그램을 종료

![경우의수]({{site.baseurl}}/images/algorithm/부분집합합.png)

```python
import sys

n = int(input())

a = list(map(int, input().split()))

total = sum(a)

def DFS(L, sum):
	if L == n : # 트리의 끝까지 탐색을 해서 인덱스가 오버되었을 때 
		if sum == (total - sum) : # 두 부분 집합이 같다면 "Yes"를 출력하고 프로그램 종료
			print("yes")
			sys.exit(0) # 아예 프로그램을 종료 -> process finished with exit code 0
		
	else : 
		DFS(L+1, sum+a[L]) # L번 인덱스의 원소를 합하겠다
		DFS(L+1, sum) # L번 인덱스의 원소를 합하지 않겠다. 
		
	

DFS(0, 0)

print("No") # 참이 되는 경우가 없을 때 함수를 재귀적으로 다 실행시키고 와서 해당 "No"를 출력
```

모든 분기를 다 찾아보지 않는 것으로 시간 복잡도를 낮출 수 있다. 

```python
import sys

n = int(input())

a = list(map(int, input().split()))

total = sum(a)

def DFS(L, sum):
	if sum > total//2 : # 가지를 더 뻗어나갈 필요가 없다. -> 시간 복잡도를 더 아낄 수 있다. 
		return 

	if L == n :
		if sum == (total - sum) : 
			print("yes")
			sys.exit(0)
		
	else : 
		DFS(L+1, sum+a[L]) # L번 인덱스의 원소를 합하겠다
		DFS(L+1, sum) # L번 인덱스의 원소를 합하지 않겠다. 
		
	

DFS(0, 0)
print("No")
```
