
# Fundamentals





## Stacks and Queues



```python
stack = []
stack.append("elem")
stack.pop()
```




    'elem'




```python
from collections import deque

queue = deque()
queue.append("elem")
queue.popleft()
```




    'elem'



## Binary Search

- Use when input is sorted
- Other cases: 
  - Sorted rotated array 
  - Median of two sorted arrays
  - Find kth element in two sorted arrays

### Basic Binary Search


```python
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
  
    while low <= high:
        mid = low + (high - low) // 2
    
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return "Not Found"
```


```python
# Test Cases
print(binary_search([1,2,4,5,7,9], 9))
print(binary_search([1,2,4,5,7,9], 101))
print(binary_search([], 12))
```

    5
    Not Found
    Not Found


### Binary Search in Sorted Rotated Array


```python
def binary_search_in_sorted_rotated(arr, target):
    low = 0
    high = len(arr) - 1
  
    while low <= high:
        mid = low + (high - low) // 2

        if arr[mid] == target:
            return mid

        if arr[low] < arr[mid]: # Left half is sorted
            if arr[low] <= target < arr[mid]:
                high = mid - 1
            else:
                low = mid + 1

        else: # Right half is sorted
            if arr[mid] < target <= arr[high]:
                low = mid + 1
            else:
                high = mid - 1    

    return "Not Found"  
```


```python
# Test Cases
print(binary_search_in_sorted_rotated([5,7,9,1,2,4], 9))
print(binary_search_in_sorted_rotated([2,4,5,7,9,1], 1))
print(binary_search_in_sorted_rotated([2,4,5,7,9,1], 2))
print(binary_search_in_sorted_rotated([2,4,5,7,9,1], 101))
print(binary_search_in_sorted_rotated([], 12))
```

    2
    5
    0
    Not Found
    Not Found


### Median of Two Sorted Arrays


```python
# If the array is even, median is the average of the two middle elements
# https://www.youtube.com/watch?v=LPFhl65R7ww
# position_x + position_y = (len(x) + len(y) + 1) // 2

# if  max_left_x <= min_right_y    and   max_left_y <= min_right_x:
  # Found

# else if  max_left_x > min_right_y:
  # Move left in x

# else:
  # Move right in x

def median_of_two_sorted_arr(arr_a, arr_b):
    x = arr_a if len(arr_a) <= len(arr_b) else arr_b # Shorter list
    y = arr_b if len(arr_a) <= len(arr_b) else arr_a # Longer list
  
    low = 0
    high = len(x) # Not len(x) - 1. We are looking at which position we can cut at
                # and that can be between two indexes
  
    while low <= high:
        x_partition_point = low + (high - low) // 2
        y_partition_point = (len(x) + len(y) + 1) // 2 - x_partition_point

        max_left_x = x[x_partition_point - 1] if x_partition_point > 0 else -float('inf')
        min_right_x = x[x_partition_point] if x_partition_point < len(x) else float('inf')
        max_left_y = y[y_partition_point - 1] if y_partition_point > 0 else -float('inf')
        min_right_y = y[y_partition_point] if y_partition_point < len(y) else float('inf')

        if  max_left_x <= min_right_y and max_left_y <= min_right_x:
            # Valid partition found, compute median     
            if (len(x) + len(y)) % 2 == 0:
                return (max(max_left_x, max_left_y) + min(min_right_x, min_right_y)) / 2
            else:
                return max(max_left_x, max_left_y)

        elif max_left_x > min_right_y:
            high = x_partition_point - 1
        else:
            low = x_partition_point + 1

    return "Not Found"
   
```


```python
print(median_of_two_sorted_arr([1, 3, 8, 9, 15],   [7, 11, 18, 19 , 21, 25]))
print(median_of_two_sorted_arr([1, 3, 8, 9],   [7, 11, 18, 19 , 21, 25]))
print(median_of_two_sorted_arr([1, 2],   [7, 11, 18, 19 , 21, 25]))
```

    11
    10.0
    14.5


### K-th Element in Two Sorted Arrays


```python
# This can be used to solve the median problem as well

def kth_element(arr_a, arr_b, k):
    if not arr_a:
        return arr_b[k]
    if not arr_b:
        return arr_a[k]
  
    a_median_index = len(arr_a) // 2
    b_median_index = len(arr_b) // 2
    
    a_median = arr_a[a_median_index]
    b_median = arr_b[b_median_index]
    
    # k is greater than the number of elements on the left of the median indexes
    if k > a_median_index + b_median_index:
        # We cam safely discard all elements before a_median
        if a_median < b_median:
            return kth_element(arr_a[a_median_index + 1:], arr_b, k - a_median_index - 1)
        else:
            return kth_element(arr_a, arr_b[b_median_index + 1:], k - b_median_index - 1)
     
    else:
        # We cam safely discard all elements after b_median
        if a_median < b_median:
            return kth_element(arr_a, arr_b[:b_median_index], k)
        else:
            return kth_element(arr_a[:a_median_index], arr_b, k)
```


```python
print(kth_element([1,2,3,4], [2,6,10,11], 0))
print(kth_element([1,2,3,4], [2,6,10,11], 1))
print(kth_element([1,2,3,4], [2,6,10,11], 2))
print(kth_element([1,2,3,4], [2,6,10,11], 3))
print(kth_element([1,2,3,4], [2,6,10,11], 4))
print(kth_element([1,2,3,4], [2,6,10,11], 5))
print(kth_element([1,2,3,4], [2,6,10,11], 6))
print(kth_element([1,2,3,4], [2,6,10,11], 7))
```

    1
    2
    2
    3
    4
    6
    10
    11


## Sorting

 | Algorithm | Stable | In-place | Worst | Average | Best |
 | ---------------- | ---------- |---------- | --------- | ------------- | -------- |
 | Selection Sort | Yes | Yes | O(n<sup>2</sup>) | O(n<sup>2</sup>) | O(n<sup>2</sup>) |
 | Bubble Sort | Yes | Yes | O(n<sup>2</sup>) | O(n<sup>2</sup>) | O(n<sup>2</sup>) |
 | Insertion Sort | Yes | Yes | O(n<sup>2</sup>) | O(n<sup>2</sup>) | O(n) |
 | Heap Sort | No | No | O(nlogn) | O(nlogn) | O(nlogn) |
 | Merge Sort | Yes | Yes | O(nlogn) | O(nlogn) | O(nlogn) |
 | Quick Sort | Yes | Yes | O(n<sup>2</sup>) | O(nlogn) | O(nlogn) |

### Selection Sort


```python
# Find the smallest and put at the first index
def selection_sort(arr):
    if not arr: return []
    min_elem = arr.pop(arr.index(min(arr)))
    return [min_elem] + selection_sort(arr)
```


```python
# Test Cases
print(selection_sort([]))
print(selection_sort([10,4,4.0,2,2,6,7,8,3,12,34]))
```

    []
    [2, 2, 3, 4, 4.0, 6, 7, 8, 10, 12, 34]


### Bubble Sort


```python
# Go through the array n times and swap two adjacent indexes if not ascending
def bubble_sort(arr):
    for j in range(len(arr)- 1, 0, -1):
        for i in range(0, j):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
    return arr
```


```python
# Test Cases
print(bubble_sort([]))
print(bubble_sort([1,4.0,4,2,2,6,7,8,3,12,34]))
```

    []
    [1, 2, 2, 3, 4.0, 4, 6, 7, 8, 12, 34]


### Insertion Sort


```python
def insertion_sort(arr):
    for i in range(1, len(arr)): 
        key, j = arr[i], i - 1
        while j >= 0 and key < arr[j] : 
            arr[j+1] = arr[j] 
            j -= 1
        arr[j+1] = key
  
    return arr
```


```python
# Test Cases
print(insertion_sort([]))
print(insertion_sort([1,4.0,4,2,2,6,7,8,3,12,34]))
```

    []
    [1, 2, 2, 3, 4.0, 4, 6, 7, 8, 12, 34]


### Heap Sort


```python
from heapq import heappop, heapify

def heap_sort(arr):
    heapify(arr)
    sorted_arr = []
    while arr:
        sorted_arr.append(heappop(arr))
    return sorted_arr
```


```python
# Test Cases
print(heap_sort([]))
print(heap_sort([1,4.0,4,2,2,6,7,8,3,12,34]))
```

    []
    [1, 2, 2, 3, 4, 4.0, 6, 7, 8, 12, 34]


### Merge Sort


```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = (len(arr) + 1) // 2 # Mid of an array is always len(arr) + 1 // 2. 
    left = arr[:mid] # NOT inclusive of mid
    right = arr[mid:] # Inclusive of mid
  
    return merge(merge_sort(left), merge_sort(right))

def merge(arr_a, arr_b):
    merged = []

    index_a = index_b = 0

    # Both arrays are non-empty
    while index_a < len(arr_a) and index_b < len(arr_b):
        if arr_a[index_a] < arr_b[index_b]:
            merged.append(arr_a[index_a])
            index_a += 1
        else:
            merged.append(arr_b[index_b])
            index_b += 1      
  
    # One of the arrays is empty
    while index_a < len(arr_a):
        merged.append(arr_a[index_a])
        index_a += 1

    while index_b < len(arr_b):
        merged.append(arr_b[index_b])
        index_b += 1

    return merged
```


```python
# Test Cases
print(merge_sort([]))
print(merge_sort([1,4.0,4,2,2,6,7,8,3,12,34]))
```

    []
    [1, 2, 2, 3, 4, 4.0, 6, 7, 8, 12, 34]


### In-place Merge Sort


```python
def merge_sort(arr):
    sort_subarray(arr, 0, len(arr) - 1)

def sort_subarray(arr, start, end):
    if end - start < 1:
        return
  
    print(arr[start:end + 1])
  
    if start >= len(arr) or end < 0:
        return 
  
    len_of_subarray = end - start + 1
    mid = ((len_of_subarray + 1) // 2) + start

    sort_subarray(arr, start, mid - 1)
    sort_subarray(arr, mid, end)
    merge(arr, start, mid - 1, mid, end)

def merge(arr, start_a, end_a, start_b, end_b):
  
    # If already sorted
    if arr[end_a] <= arr[start_b]:
        return
  
    while start_a < end_a and start_b < end_b:
        if arr[start_a] <= arr[start_b]:
            start_a += 1
        else:
            # Shift all elements accordingly
            arr.insert(arr.pop(start_b), start_a)
            start_a += 1
            start_b += 1
            end_a += 1

  
```


```python
# Test Cases
print(merge_sort([]))
r = [1,4.0,4,2,2,6,7,8,3,12,34]
print(merge_sort(r))
print(r)
```

    None
    [1, 4.0, 4, 2, 2, 6, 7, 8, 3, 12, 34]
    [1, 4.0, 4, 2, 2, 6]
    [1, 4.0, 4]
    [1, 4.0]
    [2, 2, 6]
    [2, 2]
    [7, 8, 3, 12, 34]
    [7, 8, 3]
    [7, 8]
    [12, 34]
    None
    [1, 4.0, 1, 4, 2, 6, 7, 8, 3, 12, 34]


### Quick Sort


```python
from random import randint  # randint is inclusive of start and end

def quick_sort(arr):
    if len(set(arr)) <= 1:
        return arr

    pivot = randint(0, len(arr) - 1)
    left = [elem for elem in arr if elem <= arr[pivot]]
    right = [elem for elem in arr if elem > arr[pivot]]

    return quick_sort(left) + quick_sort(right)
```


```python
# Test Cases
print(quick_sort([]))
print(quick_sort([1,4.0,4,2,2,6,7,8,3,12,34]))
```

    []
    [1, 2, 2, 3, 4.0, 4, 6, 7, 8, 12, 34]


### In-place Quick Sort


```python
def quicksort(a):
    in_place_quicksort(arr, 0, len(a) - 1)
    return arr

def swap(arr, x, y):
    temp = arr[x]
    arr[x] = arr[y]
    arr[y] = temp

def in_place_quicksort(a, start, end):
    if start >= end:
        return a

    pivot_index = random.randint(start, end)
    pivot = a[pivot_index]
    swap(a, start, pivot_index)

    left_ptr = start + 1
    right_ptr = end

    while (left_ptr < right_ptr):
        if a[left_ptr] >  pivot:
            if a[right_ptr] < pivot:
                swap(a, left_ptr, right_ptr)
                left_ptr += 1
                right_ptr -= 1
            else:
                right_ptr -= 1

        else:
            left_ptr += 1

    swap(a, left_ptr, start)
    in_place_quicksort(a, start, left_ptr - 1)
    in_place_quicksort(a, left_ptr + 1, end)
```


```python
# Test Cases
print(quick_sort([]))
print(quick_sort([1,4.0,4,2,2,6,7,8,3,12,34]))
```

    []
    [1, 2, 2, 3, 4.0, 4, 6, 7, 8, 12, 34]


### Quick Select


```python
# T(n) = T(n/2) + n
#      = T(n/4) + n/2 + n
#      = T(n/16) + n/4 + n/2 + n
#      < 2n
```

## Trees


```python
class BSTNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right= right
```


```python
# Height of a tree
def height(root):
    if not root.left and not root.right:
        return 0

    if not root.left:
        return 1 + height(root.right)

    if not root.right:
        return 1 + height(root.left)

    return 1 + max(height(root.left), height(root.right))
```


```python
# Search in a BST
def search(root, value):
    if not root:
        return "Not in tree"
    if root.val == value:
        return value
    elif root.val >= value:
        return search(root.left, value)
    else:
        return search(root.right, value)
```


```python
# Insert in BST
def insert(root, value):
    if not root:
        return Node(value)
  
    if value <= root:
        if root.left:
            insert(root.left, value)
        else:
            root.left = Node(value)
    else:
        if root.right:
            insert(root.right, value)
        else:
            root.right = Node(value)

# Make tree from array
def interative_insert(arr):
    if not arr:
        return None
  
    root = Node(arr[0])
  
    for elem in arr:
        insert(root, elem)
  
    return root
```


```python
# Successor

# Left most element of the right subtree
# If no right child, climb up the parent tree until you turn right
```


```python
# Delete

# No child: Delete
# One child: Join child to parent
# Two children: 1. Swap with successor
#               2. Delete node (Successor has at most one child)
```

## Graphs


```python
# Graphs as adjacency list
graph = {'A': ['B', 'F', 'G'],
         'B': ['E', 'D', 'A'],
         'C': ['E', 'F'],
         'D': ['A', 'D', 'B'],
         'E': ['C', 'F', 'A'],
         'F': ['B', 'D', 'G'],
         'G': ['C', 'F', 'A']
        }

graph2 = {'A': ['C', 'B'],
          'B': ['D'],
          'C': [],
          'D': []
         }

graph3 = {'A': ['B', 'C', 'D'],
          'B': ['E'],
          'C': [],
          'D': ['E'],
          'E': ['B', 'D']
         }
```

### BFS

If you have to set and deal with parent pointers, use this


```python
from collections import deque

def BFS(graph, source):
    queue = deque()
    visited = set()

    queue.append(source)
    visited.add(source)
  
    while queue:
        current_node = queue.popleft()

        print(current_node)
    
        for neighbour in graph[current_node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

BFS(graph2, 'A')
```

    A
    C
    B
    D


```python
def bfs_with_path_tracking(graph, source, target):
    queue, visited, parent = deque(), set(), dict()
    
    queue.append(source)
    visited.add(source)
    parent[source] = source
    
    while queue:
        current_node = queue.popleft()
        
        if current_node == target:
            break
        
        for neighbour in graph[current_node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
                parent[neighbour] = current_node
    
    if target not in parent:
        raise ValueError(f"Target {target} not found")
    
    path = deque(target)
    while path[0] != source:
        path.appendleft(parent[path[0]])
        
    return list(path)
    
                
print(bfs_with_path_tracking(graph3, 'A', 'E'))
```

    ['A', 'B', 'E']


### DFS


```python
def DFS(graph, source):
    visited = set()
  
    def DFS_helper(graph, source):
        visited.add(source)

        print(source)
    
        for neighbour in graph[source]:
            if neighbour not in visited:
                DFS_helper(graph, neighbour)
  
    DFS_helper(graph, source)

DFS(graph3, 'A')
```

    A
    B
    E
    D
    C



```python
def DFS(graph, source):
    visited = set()

    def DFS_helper(graph, source):
        if source in visited:
            return

        visited.add(source)

        print(source)

        for neighbour in graph[source]:
            DFS_helper(graph, neighbour)

    DFS_helper(graph, source)


DFS(graph3, 'A')
```

    A
    B
    E
    D
    C



```python
def dfs_all_paths(graph, source, target):
    visited = set()
    paths = []
    
    def dfs_helper(path):
        last_node = path[-1]
        
        if last_node in visited:
            return
        
        if last_node == target:
            paths.append(path)
            return
            
        
        visited.add(last_node)
        
        for neighbour in graph[last_node]:
            dfs_helper(path + [neighbour])
        
        visited.remove(last_node)
    
    dfs_helper([source])
    
    return paths


dfs_all_paths(graph3, 'A', 'D')
```
    [['A', 'B', 'E', 'D'], ['A', 'D']]

### Incorrect DFS (replacing queue with stack)


```python
def incorrect_DFS(graph, source):
    stack = deque()
    visited = set()

    stack.append(source)
    visited.add(source)

    while stack:
        current_node = stack.pop()

        print(current_node)

        for neighbour in graph[current_node]:
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append(neighbour)


incorrect_DFS(graph3, 'A')  # The answer should have been A D E B C or A B E D C
```

    A
    D
    E
    C
    B


### BFS and DFS with Queue <> Stack

In a normal BFS, a node is marked as visited right after being added to the queue. If in this implementation, the queue had to replaced with a stack, it doesn't become DFS. Here's an example (this is graph 3 from above):

<img width="300px" src="https://user-images.githubusercontent.com/23443586/61193224-f182b800-a66e-11e9-980d-7dbee4c6dfaa.png">

In the implementation below, a node is marked as visited only when it is popped from the queue. In this implementation, interchanging the queue witha stack brings us from a BFS to a DFS.


```python
from collections import deque


def BFS(graph, source):
    queue = deque()
    visited = set()
    queue.append(source)

    while (len(queue) != 0):
        current_node = queue.popleft()

        if not current_node in visited:
            print(current_node)
            visited.add(current_node)
            for neighbour in graph[current_node]:
                queue.append(neighbour)


BFS(graph3, 'A')
```

    A
    B
    C
    D
    E



```python
def DFS(graph, source):
    stack = deque()
    visited = set()
    stack.append(source)

    while (len(stack) != 0):
        current_node = stack.pop()

        if not current_node in visited:
            print(current_node)
            visited.add(current_node)
            for neighbour in graph[current_node]:
                stack.append(neighbour)

DFS(graph3, 'A')
```

    A
    D
    E
    B
    C


## Union Find Data Structure

Two methods:
1. Union
2. Find

Optimsations:
1. Path Compression: When executing Find, all the child and grandchild nodes you encounter, set their parent to be the top-most parent
2. Rank-based union: When unioning two cliques, put the one with the lower rank below the one with a higher rank

<img width="800" src="https://user-images.githubusercontent.com/23443586/61196706-5b0dc100-a685-11e9-8a2c-4f5da2af600f.png">


```python
# Basic implementations
def find(x):
    if parent[x] == x:
        return parent[x]
    return find(parent[x])


def union(x, y):
    xroot, yroot = find(x), find(y)
    if xroot == yroot:
        return
    parent[xroot] = yroot
```


```python
# With optimisations
def find(x):
    if parent[x] == x:
        return x
    root = find(parent[x])
    parent[x] = root


def union(x, y):
    xroot, yroot = find(x), find(y)
    if xroot == yroot:
        return

    if rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    elif rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1
```
