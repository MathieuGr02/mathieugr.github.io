---
title: Sorting
layout: collection
permalink: /Algorithms-Datastructures/Sorting
collection: Algorithms-Datastructures
entries_layout: grid
mathjax: true
toc: true
categories:
  - study
tags:
  - programming
---


```python
import numpy as np
```

# Sorting

Imagine we have a sequence of $n$ elements $e_1, ..., e_n$, where each element has a key $k_i = key(e_i)$.
In sorting we impose a partial order onto the keys which then have the following properties
* Reflexive: $k \leq k$
* Transitive: $k \leq k'$ and $k' \leq k'' \Rightarrow k \leq k''$
* Antisymmetric: $k \leq k'$ and $k' \leq k \Rightarrow k = k'$

# Analysis

* ***Running Time***: How many key comparisons and swaps of elements are executed
* ***Space Consumption***: How much space is used in addition to the space occupied by the input sequence

# Properties

* ***Adaptive***: A algorithm is adaptive if it is faster on an already partially sorted input compared to an unsorted one
* ***In-place***: A algorithm is in-place if it needs no additional storage beyonf the input array and a constant amount of space, i.e. its space consumption is independent of the input size 
* ***Stable***: A algorithm is stable if when there appear elements with the same value, then in the output sequence they hold the same order as in the input sequence

# Selection Sort

---

Selection sort works by iterating over the whole array from the i-th position, selecting the smallest value in the range $[i, n]$ and swapping it with the $i$-th element and then increment $i+1 \rightarrow i$.

### Properties:

- Not stable => Order of elements can be reversed during swapping
-  In-place => Doesn't need extra storage
-  Not adaptive => Still iterates over whole array to find maximum, so fully sorted array still has same running time

### Running time:

$$\Omega(n^2) \ \text{and} \ O(n^2) \Rightarrow \Theta(n^2) = \{ f | \exists c > 0 \exists c' > 0 \exists n_0 > 0 \forall n \geq n_0: cn^2 \leq f(n)\leq c'n^2 \}$$

### Code:


```python
def selection_sort(array: list) -> None:
    n = len(array)
    for i in range(n - 1):
        min_index = i
        for j in range(i + 1, n):
            if array[j] < array[min_index]:
                min_index = j
        array[i], array[min_index] = array[min_index], array[i]
```




```python
a = [4, 5, 2, 3, 1, 9, 11, 4, 10, 8]
print(f'Before sorting: {a}')
selection_sort(a)
print(f'Selection sort: {a}')
```

    Before sorting: [4, 5, 2, 3, 1, 9, 11, 4, 10, 8]
    Selection sort: [1, 2, 3, 4, 4, 5, 8, 9, 10, 11]
    

# Insertion Sort

---

Insertion sort works by taking the element at the i'th position and swapping it with its previous element until it is correctly placed in the sequence $[0, i]$, which is then sorted.

### Properties:
- Stable => Only swaps if larger thus elements stay in their order
- In-place => Doesnt require extra storage
- Adaptive => While loop doesn't run if array is already sorted

### Running time:
$$
\Omega(n) \ \text{and} \ O(n^2)
$$

### Code:


```python
def insertion_sort_1(array: list) -> None:
    for i in range(1, len(array)):
        min_index = i
        while min_index > 0 and array[min_index] < array[min_index - 1]:
            array[min_index], array[min_index - 1] = array[min_index - 1], array[min_index]
            min_index -= 1

def insertion_sort_2(array: list) -> None:
    for i in range(1, len(array)):
        val = array[i]
        index = i
        while index > 0 and val < array[index - 1]:
            array[index] = array[index - 1]
            index -= 1
        array[index] = val 
```


```python
a = [4, 5, 2, 3, 1, 9, 11, 4, 10, 8]
b = [4, 5, 2, 3, 1, 9, 11, 4, 10, 8]
print(f'Before sorting: {a}')
insertion_sort_1(a)
insertion_sort_2(b)
print(f'Insertion sort 1: {a}')
print(f'Insertion sort 2: {b}')
```

    Before sorting: [4, 5, 2, 3, 1, 9, 11, 4, 10, 8]
    Insertion sort 1: [1, 2, 3, 4, 4, 5, 8, 9, 10, 11]
    Insertion sort 2: [1, 2, 3, 4, 4, 5, 8, 9, 10, 11]
    

# Merge Sort

---

Merge sort works by recursively calling the sorting on the array until every recursion list has 1 element, and then combining and rearranging the elements with the next recursion list until the whole list is sorted.

### Properties:
- Stable => Because we check take the left element if '$\leq$' so it's order is preserved
- Not in-place => Temp array is proportional to the length of the array
- Adaptive => Standard Merge sort is not adaptive

### Running time:
In total it takes:
$$
\Omega(n \log_2 n ) \ \text{and} \ O(n \log_2 n) \Rightarrow \Theta(n \log_2 n) = \{ f | \exists c > 0 \exists c' > 0 \exists n_0 > 0 \forall n \geq n_0:  
c n \log_2 n \leq f(n) \leq c' n \log_2 n
\}
$$

And the merge step itself is:
$$
\Omega(n) \ \text{and} \ O(n) \Rightarrow \Theta(n) = \{ f | \exists c > 0 \exists c' > 0 \exists n_0 > 0 \forall n \geq n_0:  
c n \leq f(n) \leq c' n
\}
$$

### Code:

## (Top down):


```python
def merge_sort_top_down(array: list, tmp: list, lo: int, hi: int) -> None:
    if lo >= hi:
        return 
    
    mid = lo + (hi - lo) // 2
    merge_sort_top_down(array, tmp, lo, mid)
    merge_sort_top_down(array, tmp, mid + 1, hi)
    merge(array, tmp, lo, mid, hi)

def merge(array: list, tmp: list, lo: int, mid: int, hi: int) -> None:
    i = lo
    j = mid + 1
    for k in range(lo, hi + 1):
        if j > hi or (array[i] <= array[j] and i <= mid):
            tmp[k] = array[i]
            i += 1
        else:
            tmp[k] = array[j]
            j += 1
    print(tmp)
    for k in range(lo, hi + 1):
        array[k] = tmp[k]
```


```python
a = [4, 5, 2, 3, 1, 9, 11, 4, 10, 8]
tmp = [0] * len(a)
print(f'Before sorting: {a}')
merge_sort_top_down(a, tmp, 0, len(a) - 1)
print(f'Merge Sort Top Down : {a}')
```

    Before sorting: [4, 5, 2, 3, 1, 9, 11, 4, 10, 8]
    [4, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    [2, 4, 5, 0, 0, 0, 0, 0, 0, 0]
    [2, 4, 5, 1, 3, 0, 0, 0, 0, 0]
    [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
    [1, 2, 3, 4, 5, 9, 11, 0, 0, 0]
    [1, 2, 3, 4, 5, 4, 9, 11, 0, 0]
    [1, 2, 3, 4, 5, 4, 9, 11, 8, 10]
    [1, 2, 3, 4, 5, 4, 8, 9, 10, 11]
    [1, 2, 3, 4, 4, 5, 8, 9, 10, 11]
    Merge Sort Top Down : [1, 2, 3, 4, 4, 5, 8, 9, 10, 11]
    

## (Bottom up):


```python
def merge_sort_bottom_up(array: list):
    n = len(array)
    temp = list(array)
    length = 1
    while length < n:
        lo = 0
        while lo < n - length:
            mid = lo + length - 1
            hi = min(lo + 2 * length - 1, n - 1)
            merge(array, tmp, lo, mid, hi)
            lo += 2 * length
        length *= 2
```


```python
a = [4, 5, 2, 3, 1, 9, 11, 4, 10, 8]
tmp = [0] * len(a)
print(f'Before sorting: {a}')
merge_sort_bottom_up(a)
print(f'Merge Sort Bottom Up: {a}')
```

    Before sorting: [4, 5, 2, 3, 1, 9, 11, 4, 10, 8]
    [4, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    [4, 5, 2, 3, 0, 0, 0, 0, 0, 0]
    [4, 5, 2, 3, 1, 9, 0, 0, 0, 0]
    [4, 5, 2, 3, 1, 9, 4, 11, 0, 0]
    [4, 5, 2, 3, 1, 9, 4, 11, 8, 10]
    [2, 3, 4, 5, 1, 9, 4, 11, 8, 10]
    [2, 3, 4, 5, 1, 4, 9, 11, 8, 10]
    [1, 2, 3, 4, 4, 5, 9, 11, 8, 10]
    [1, 2, 3, 4, 4, 5, 8, 9, 10, 11]
    Merge Sort Bottom Up: [1, 2, 3, 4, 4, 5, 8, 9, 10, 11]
    

# Quicksort

---

Quick sort works by finding a pivot (randomly is most preffered), swapping that element to the beginning of the recursion list and rearraning the recursion list such that on the left of the pivot there are only elements smaller or equal to it and on the right there are only elements larger or equal to it. Then we recursively call the left and right side of the pivot and do the same.

### Properties:
- Not stable => pivot gets swapped and may change ordering
- Not in-place => Recursion needs memory
- Not Adaptive => Still runs to total iterations


### Running time:
$$
\Omega(n \log_2 n) \ \text{and} \ O(n^2)
$$

### Code:


```python
def quicksort(array: list, lo: int, hi: int) -> None:
    if hi <= lo:
        return
    pivot = np.random.randint(lo, hi)
    array[lo], array[pivot] = array[pivot], array[lo]
    
    pivot_pos = partition(array, lo, hi)
    quicksort(array, lo, pivot_pos - 1)
    quicksort(array, pivot_pos + 1, hi)
    
def partition(array, lo, hi) -> int:
    i = lo + 1
    j = hi
    
    for _ in range(hi - lo):
        while array[i] < array[lo] and i < hi:
            i += 1
        while array[j] > array[lo]:
            j -= 1
        if i >= j:
            break
        
        array[i], array[j] = array[j], array[i]
        i += 1
        j -= 1
    array[lo], array[j] = array[j], array[lo]
    return j
```


```python
a = [4, 5, 2, 3, 1, 9, 11, 4, 10, 8]
print(f'Before sorting: {a}')
quicksort(a, 0, len(a) - 1)
print(f'Quicksort: {a}')
```

    Before sorting: [4, 5, 2, 3, 1, 9, 11, 4, 10, 8]
    Quicksort: [1, 2, 3, 4, 4, 5, 8, 9, 10, 11]
    

# Counting sort

---

Counting sort works by counting the number of occurences of a number i into an array at the i'th position and then creating a new array and append the amount of times each element occured. 

### Properties:
- Not stable => Counting does not keep arangement
- Not in-place => counts list requires extra non constant storage
- Not adaptive  

### Running time:
If we have n elements in the array and a max value k:

$$
\Theta(n + (k + 1) + n) = \Theta(n + k) = \{ f | \exists c > 0 \exists c' > 0 \exists n_0 > 0 \forall n \geq n_0:  
c (n + k) \leq f(n) \leq c' (n + k)
\}
$$

### Code:



```python
def counting_sort(array: list, max: int) -> None:
    count = [0] * max
    for element in array:
        count[element] += 1
    
    i = 0
    for number in range(max):
        for amount in range(count[number]):
            array[i] = number
            i += 1
```


```python
a = [4, 5, 2, 3, 1, 9, 11, 4, 10, 8]
print(f'Before sorting: {a}')
counting_sort(a, max(a) + 1)
print(f'Counting sort: {a}')
```

    Before sorting: [4, 5, 2, 3, 1, 9, 11, 4, 10, 8]
    Counting sort: [1, 2, 3, 4, 4, 5, 8, 9, 10, 11]
    

# Radix Sort

---

Radix sort works by placing numbers in buckets corresponding to their i'th digit, repeating this process for the amount of digits the largest number has gives a sorted list.

### Properties:
- Non comperative => Placement in buckts doesnt require if statements
- Not in-place => Buckets filled with n extra digits
- Stable => First element always gets added to buckets and then to the list first
- Not adaptive

### Running time:
If m is the amount of digits in the largest value:
$$
\Theta(m * (b + n + b + n)) = \Theta(m(b + n)) = \{ f | \exists c > 0 \exists c' > 0 \exists n_0 > 0 \forall n \geq n_0:  
c m(b + n) \leq f(n) \leq c' m(b + n)
\}
$$

### Code:


```python
def radix_sort(array: list, base = 10):
    max_value = max(array)
    n = len(array)
    iteration = 0
    while base ** iteration <= max_value:
        buckets =  [[] for _ in range(base)]
        for i in range(n):
            buckets[array[i] // (base ** iteration) % base].append(array[i])
        
        j = 0
        for bucket in buckets:
            for element in bucket:
                array[j] = element
                j += 1
        iteration += 1
```


```python
a = [14, 5, 652, 43, 1, 29, 121, 4, 120, 81]
print(f'Before sorting: {a}')
radix_sort(a)
print(f'Radix Sort: {a}')
```

    Before sorting: [14, 5, 652, 43, 1, 29, 121, 4, 120, 81]
    Radix Sort: [1, 4, 5, 14, 29, 43, 81, 120, 121, 652]
    

# Heap sort




# Summary

---

| Algorithm      | Running complexity<br/> (best/avg./worst) | Space complexity | stable | 
|----------------|:------------------------------------------|------------------|--------|
| Selection sort | $n^2$                                     | 1                | No     | 
| Insertion sort | $n/n^2/n^2$                               | 1                | Yes    | 
| Merge sort     | $n \log_2 n$                              | n                | Yes    |
| Quicksort      | $n \log_2 n/n \log_2 n/n^2$               | $\log_2 n$       | No     |
| Counting sort  | $n + k$                                   | k                | No     |
| Radix sort     | $m(b + n)$                                | n + b            | Yes    |
| Heap sort      | $n \log_2 n$                              | 1                | No     | 


# Comparison based algorithms

---

Consider a binary tree with a root, nodes and each node having two successors. Node without a successor are called leaves. The binary tree reaches a maximum depth of atleast $\log_2 k$ with $k$ leaves.

It then holds that every comparison based sorting algorithm requires $\Omega(n \log_2 n)$ key comparisons. Thus the lower bound for sorting is $\Omega(n \log_2 n)$.
