---
toc: true
layout: post
comments: true
title: 1이 될 때까지
description: 1이 될 때까지 - 그리디 문제 (이코테 2021)
categories: [algorithm]
image:
---
## [문제]
어떠한 수 N이 1이 될 때까지 다음의 두 과정 중 하나를 반복적으로 선택하여 수행하려고 한다. 
단, 두 번째 연산은 N이 K로 나누어떨어질 때만 선택할 수 있다.

 

1. N에서 1을 뺀다.
2. N을 K로 나눈다.
예를 들어 N이 17, K가 4라고 가정하자. 이때 1번의 과정을 한 번 수행하면 N은 16이 된다. 이후에 2번의 과정을 두 번

 

수행하면 N은 1이 된다. 결과적으로 이 경우 전체 과정을 실행한 횟수는 3이 된다. 이는 N을 1로 만드는 최소 횟수이다.

 

N과 K가 주어질 때 N이 1이 될 때까지 1번 혹은 2번의 과정을 수행해야 하는 최소 횟수를 구하는 프로그램을 작성하시오.

```python
<입력 예시>
25 5
<출력 예시>
2
<입력 예시>
17 4
<출력 예시>
3
```

#### 내 풀이

```python
n, k = map(int, input().split())

# 26 -> 5 -> 1
cnt = 0
flag = True
while( n%k != 0 ) :
    n -= 1
    cnt += 1
    print(f'{n} : {cnt}')

while(flag) :
    print(f'{n} : {cnt}')
    n = n//k
    cnt += 1
    if n == 1 :
        print(f'{n} : {cnt}')
        flag = False

print(cnt)

```

```python
25 5
25 : 0
5 : 1
1 : 2
2

-----

17 4
16 : 1
16 : 1
4 : 2
1 : 3
3

```

아래의 코드는 시간복잡도가 $$log(n)$$ 이 나오게 된다.
```python
n, k = map(int, input().split())

result = 0

while True:
    # n 이 k로 나누어 떨어지는 수가 될 때까지 빼기
    target = (n // k) * k

    # 마지막 n이 1일 때 0 을 빼서 1이 더해짐 => 반복문을 빠져나와서 1을 빼줘야함
    result += (n - target)

    n = target

    print(f'n :{n} result : {result} target : {target}')

    # n이 k보다 작을 때 반복문 탈출 -> 더 이상 나눌 수 없음
    if n < k:
        break

    # k로 나누기
    result += 1
    n //= k

    print(f'n :{n} result : {result} target : {target}')

# 마지막으로 남은 수에 대하여 1씩 빼기
result += (n - 1)

print(f'n :{n} result : {result} target : {target}')

print(result)

```