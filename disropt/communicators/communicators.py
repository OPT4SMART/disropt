from mpi4py import MPI
import dill


class Communicator():
    """Communicator abstract class

    """

    def __init__(self):
        pass

    def send(self):
        pass

    def receive(self):
        pass

    def neighbors_send(self, obj, neighbors):
        pass

    def neighbors_receive(self, neighbors):
        pass

    def neighbors_receive_asynchronous(self, neighbors):
        pass

    def neighbors_exchange(self, send_obj, in_neighbors, out_neighbors, dict_neigh):
        pass


class MPICommunicator(Communicator):
    """MPICommunicator class performs communications through MPI. Requires mpi4py.

    Attributes:
        comm: communication world
        size (int): size of the network. 
        rank (int): rank of the processor

    """

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.size = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()

    def neighbors_send(self, obj, neighbors):
        """Send data to neighbors

        Args:
            obj (Any): object to send
            neighbors (list): list of neighbors
        """
        obj_send = dill.dumps(obj)
        for neighbor in neighbors:
            req = self.comm.Isend(obj_send, dest=neighbor, tag=neighbor)
            self.requests.append(req)

    def neighbors_receive(self, neighbors):
        """Receive data from neighbors (waits until data are received from all neighbors)

        Args:
            neighbors (list): list of neighbors

        Returns:
            dict: dict containing received data associated to each neighbor in neighbors
        """
        received_data = {}
        count = 0
        while(count < len(neighbors)):
            state = MPI.Status()
            okay = self.comm.Iprobe(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=state)
            if(okay):
                node = state.Get_source()
                data = bytearray(state.Get_count())
                self.comm.Recv(data, source=node, tag=state.Get_tag())
                data = dill.loads(data)
                if node not in received_data:
                    count += 1
                received_data[node] = data
            else:
                pass
        return received_data

    def neighbors_receive_asynchronous(self, neighbors):
        """Receive data (if any) from neighbors.

        Args:
            neighbors (list): list of neighbors

        Returns:
            dict: dict containing received data
        """
        received_data = {}
        state = MPI.Status()
        while self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=state):
            node = state.Get_source()
            data = self.comm.recv(source=node)
            received_data[node] = data
            state = MPI.Status()

        return received_data

    def neighbors_exchange(self, send_obj, in_neighbors, out_neighbors, dict_neigh):
        """exchange information (synchronously) with neighbors

        Args:
            send_obj (any): object to send
            in_neighbors (list): list of in-neighbors
            out_neighbors (list): list of out-neighbors
            dict_neigh: True if send_obj contains a dictionary with different objects for each neighbor

        Returns:
            dict: dict containing received data associated to each neighbor in in-neighbors
        """
        # self.comm.Barrier()

        self.requests = []
        if dict_neigh == False:
            self.neighbors_send(send_obj, out_neighbors)
        else:
            for j in out_neighbors:
                self.neighbors_send(send_obj[j], [j])
        data = self.neighbors_receive(in_neighbors)
        # MPI.Request.Waitall(self.requests)
        self.comm.Barrier()
        return data
