def insertSort(array):
    for i in range(1, len(array)):
        if(array[i-1] > array[i]):
            temp = array[i]
            index = i
            while(index > 0 and array[index - 1] > temp):
                array[index] = array[index - 1]
                index -= 1
            array[index] = temp

def halfInsertSort(array):
    sentinel = 0
    for i in range(1, len(array)):
        if (array[i - 1] > array[i]):
            temp = array[i]
            low = 0
            high = i - 1
            while(low <= high):
                mid = (low + high) >> 1
                if(temp < array[mid]):
                    high = mid - 1
                else:
                    low = mid + 1
            for j in range(i, low, -1):
                array[j] = array[j - 1]
            array[low] = temp

proba_list = [0, 5, 10, 8, 100, 50, -10, 6]
insertSort(proba_list)
print(proba_list)

if -1:
    proba_list = [0, 5, 10, 8, 100, 50, -10, 6]
    halfInsertSort(proba_list)
    print(proba_list)