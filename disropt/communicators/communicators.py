from mpi4py import MPI
from threading import Event
from typing import List, Dict, Any
import dill


class Communicator():
    """Communicator abstract class

    """

    def __init__(self):
        pass

    def neighbors_send(self, obj: Any, neighbors: List[int]):
        pass

    def neighbors_receive(self, neighbors: List[int], stop_event: Event) -> Dict[int, Any]:
        pass

    def neighbors_receive_asynchronous(self, neighbors: List[int]) -> Dict[int, Any]:
        pass

    def neighbors_exchange(self, send_obj: Any, in_neighbors: List[int], out_neighbors: List[int],
        dict_neigh: bool, stop_event: Event) -> Dict[int, Any]:
        pass


class MPICommunicator(Communicator):
    """Communicator class that performs communications through MPI. Requires mpi4py.

    Attributes:
        comm: communication world
        size (int): size of the network. 
        rank (int): rank of the processor

    """

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.size = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.requests = []

    def neighbors_send(self, obj: Any, neighbors: List[int]):
        """Send data to neighbors.

        Args:
            obj: object to send
            neighbors: list of out-neighbors
        """
        obj_send = dill.dumps(obj)
        for neighbor in neighbors:
            req = self.comm.Isend(obj_send, dest=neighbor, tag=neighbor)
            self.requests.append(req) # TODO in case this function is called directly, this list may explode

    def neighbors_receive(self, neighbors: List[int], stop_event: Event = None) -> Dict[int, Any]:
        """Receive data from neighbors (waits until data are received from all neighbors).

        Args:
            neighbors: list of in-neighbors
            stop_event: an Event object that is monitored during the execution.
                If the event is set, the function returns immediately.
                Defaults to None (does not wait upon any event)

        Returns:
            data received by in-neighbors
        """
        received_data = {}

        while(len(received_data) < len(neighbors)):

            if stop_event is not None and stop_event.is_set():
                break

            # cycle over remaining neighbors
            for node in [k for k in neighbors if k not in received_data]:
                state = MPI.Status()
                okay = self.comm.Iprobe(source=node, tag=MPI.ANY_TAG, status=state)

                if(okay):
                    data = bytearray(state.Get_count())
                    self.comm.Recv(data, source=node, tag=state.Get_tag())
                    data = dill.loads(data)
                    received_data[node] = data

        return received_data

    def neighbors_receive_asynchronous(self, neighbors: List[int]) -> Dict[int, Any]:
        """Receive data (if any) from neighbors.

        Args:
            neighbors: list of in-neighbors

        Returns:
            data received by in-neighbors (if any)
        """
        received_data = {}
        state = MPI.Status()
        while self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=state):
            node = state.Get_source()
            data = self.comm.recv(source=node)
            received_data[node] = data
            state = MPI.Status()

        return received_data

    def neighbors_exchange(self, send_obj: Any, in_neighbors: List[int], out_neighbors: List[int],
            dict_neigh: bool=False, stop_event: Event = None) -> Dict[int, Any]:
        """Exchange information (synchronously) with neighbors.

        Args:
            send_obj: object to send
            in_neighbors: list of in-neighbors
            out_neighbors: list of out-neighbors
            dict_neigh: True if send_obj contains a dictionary with different objects for each neighbor.
                Defaults to False
            stop_event: an Event object that is monitored during the execution.
                If the event is set, the function returns immediately.
                Defaults to None (does not wait upon any event)

        Returns:
            data received by in-neighbors
        """

        self.requests = []
        if dict_neigh == False:
            self.neighbors_send(send_obj, out_neighbors)
        else:
            for j in out_neighbors:
                self.neighbors_send(send_obj[j], [j])
        data = self.neighbors_receive(in_neighbors, stop_event)
        MPI.Request.Waitall(self.requests)
        # self.comm.Barrier()
        return data
