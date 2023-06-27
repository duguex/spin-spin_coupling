from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# if rank == 0:
#     data = {'a': 7, 'b': 3.14}
#     comm.send(data, dest=1)
# if rank == 1:
#     data = comm.recv(source=0)
#     print('On process 1, data is ', data)

# if rank == 0:
#     data = {'key1': [1, 2, 3],
#             'key2': ('abc', 'xyz')}
# else:
#     data = None
#
# data = comm.bcast(data, root=0)
# print('Rank: ', rank, ', data: ', data)

# numDataPerRank = 10
# data = None
# if rank == 0:
#     data = np.linspace(1, size * numDataPerRank, size * numDataPerRank)
#
# recvbuf = np.empty(numDataPerRank, dtype='d')  # allocate space for recvbuf
# comm.Scatter(data, recvbuf, root=0)
#
# print('Rank: ', rank, ', recvbuf received: ', recvbuf)


# if rank == 0:
#     send_obj = 0.5
# elif rank == 1:
#     send_obj = 1.5
# elif rank == 2:
#     send_obj = 2.5
# else:
#     send_obj = 3.5
#
# print(1, rank, comm.reduce(send_obj, op=MPI.SUM, root=1))
# print(2, rank, comm.reduce(send_obj, op=MPI.MAX, root=2))
# print(3, rank, comm.reduce(send_obj, op=MPI.SUM, root=3))

# send_buf = np.array([0, 1], dtype='i') + 2 * rank
# recv_buf = np.empty(2, dtype='i') if rank == 2 else None
# comm.Reduce(send_buf, recv_buf, op=MPI.SUM, root=2)
# print(rank, send_buf, recv_buf)


# if rank == 2:
#     recv_buf = np.zeros(2, dtype='i') - 1
# else:
#     send_buf = np.array([0, 1], dtype='i') + 2 * rank
#     recv_buf = None
#
# if rank == 2:
#     comm.Reduce(MPI.IN_PLACE, recv_buf, op=MPI.SUM, root=2)
# else:
#     comm.Reduce(send_buf, recv_buf, op=MPI.SUM, root=2)
#
# print(rank, recv_buf)


# send_obj = [i + 0.5 for i in range(8)][rank]
# print("scan&exscan", rank, comm.scan(send_obj), comm.exscan(send_obj))

# send_buf = np.array([0, 1], dtype='i') + rank
# recv_buf = np.empty(2, dtype='i')
# comm.Scan(send_buf, recv_buf, op=MPI.SUM)
# print("scan_np", rank, recv_buf)

# send_buf = np.array([0, 1], dtype='i') + rank
# recv_buf = np.zeros(2, dtype='i') - 1
# comm.Exscan(send_buf, recv_buf, op=MPI.SUM)
# print("exscan_np", rank, recv_buf)

# recv_buf = np.array([0, 1], dtype='i') + rank
# comm.Scan(MPI.IN_PLACE, recv_buf, op=MPI.SUM)
# print("scan_np_in_place", rank, recv_buf)
#
# recv_buf = np.array([0, 1], dtype='i') + 2 * rank
# comm.Exscan(MPI.IN_PLACE, recv_buf, op=MPI.SUM)
# print("exscan_np_in_place", rank, recv_buf)
