import numpy as np

file1 = open('d_tough_choices.txt', 'r')
Lines = file1.readlines()

k, n, time_lib = [int(x) for x in Lines[0].split()]         #num of books, number of libraries, time for scan
kd_vec = np.array([int(x) for x in Lines[1].split()])

file_mtx = np.zeros(n*k).reshape(n,k).astype(np.int32)

st_vec = []
kos_vec = []
book_num_vec = []

for i, line in enumerate(Lines[2:]):
    if i % 2:   #tek book_ids
        int_cast = tuple([int(x) for x in line.split()])
        file_mtx[int((i-1)/2), int_cast] = 1

    else:   #cift library stamp
        b, sup, bpd = [int(x) for x in line.split()]
        st_vec.append(sup)
        kos_vec.append(bpd)
        book_num_vec.append(b)

book_map = np.concatenate((np.arange(k).reshape(1, k), kd_vec.reshape(1, k)), axis=0)
book_map = book_map[:, book_map[1, :].argsort()]

file_mtx = np.concatenate((kd_vec.reshape(1, k), file_mtx), axis=0)
file_mtx = file_mtx[:, file_mtx[0, :].argsort()][1:,:]

st_vec = np.array(st_vec).astype(np.int32)
kos_vec = np.array(kos_vec).astype(np.int32)
book_num_vec = np.array(book_num_vec).astype(np.int32)

day = 0
counter = 0
output = []
while (day < time_lib) and (counter < n):
    print(day)
    time_available_to_read = np.maximum(0, ((time_lib - day) - st_vec))

    x_book_vec = np.minimum(np.multiply(kos_vec, time_available_to_read, dtype=np.int64), book_num_vec)

    for i in range(n):
        if x_book_vec[i]:
            leverage = np.where(file_mtx[i,:])[0][x_book_vec[i]-1]  #index of the last book can be read
            file_mtx[i, leverage+1:] = 0    #clear the book cannot be read

    lib_profit_vec = np.dot(file_mtx, book_map[1,:])
    best_lib = lib_profit_vec.argmax() #choosen library id

    day += st_vec[best_lib]
    books = np.where(file_mtx[best_lib, :])[0] #will be scanned books from best_lib

    output.append([[best_lib, books.size], book_map[0, books]])

    file_mtx[best_lib,:] = 0    #clear best_lib
    file_mtx[:, books] = 0  #clear books which is already read

    book_num_vec = np.count_nonzero(file_mtx, axis=1)
    counter +=1


out_file = open("d_out.txt", "w")
out_file.write(str(counter)+"\n")

for l in range(counter):
    out_file.write(str(output[l][0][0]) + " " + str(output[l][0][1]) + "\n")

    listToStr = ''.join([str(book)  if i == 0 else " " + str(book) for i, book in enumerate(output[l][1])])

    out_file.write(listToStr + "\n")

out_file.close()
file1.close()
