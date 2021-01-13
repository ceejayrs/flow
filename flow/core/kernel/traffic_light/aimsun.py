"""Script containing the base traffic light kernel class."""
from flow.core.kernel.traffic_light.base import KernelTrafficLight


class AimsunKernelTrafficLight(KernelTrafficLight):
    """Aimsun traffic light kernel.

    Implements all methods discussed in the base traffic light kernel class.
    """

    def __init__(self, master_kernel):
        """Instantiate the sumo traffic light kernel.

        Parameters
        ----------
        master_kernel : flow.core.kernel.Kernel
            the higher level kernel (used to call methods from other
            sub-kernels)
        """
        KernelTrafficLight.__init__(self, master_kernel)

        # names of nodes with traffic lights
        self.__ids = []
        self.num_meters = 0

    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api

    def update(self, reset):
        """See parent class."""
        pass

    def get_ids(self):
        """See parent class."""
        return self.kernel_api.get_traffic_light_ids()

    def set_state(self, meter_aimsun_id, state):
        """Set the state of the traffic lights on a specific meter.

        Parameters
        ----------
        meter_aimsun_id : int
            aimsun id of the meter
        state : int
            desired state(s) for the traffic light
            0: red
            1: green
            2: yellow
        link_index : int, optional
            index of the link whose traffic light state is meant to be changed.
            If no value is provided, the lights on all links are updated.
        """
        self.kernel_api.set_traffic_light_state(meter_aimsun_id, None, state)

    def get_state(self, meter_aimsun_id):
        """Return the state of the traffic light(s) at the specified node.

        Parameters
        ----------
        meter_aimsun_id: int
            aimsun id of the meter

        Returns
        -------
        state : int
            desired state(s) for the traffic light
            0: red
            1: green
            2: yellow
        """
        return self.kernel_api.get_traffic_light_state(meter_aimsun_id)

    def get_intersection_offset(self, node_id):
        """
        Gets the intersection's offset

        Parameters
        ----------
        node_id : int
            the node id of the intersection

        Returns
        -------
        int
            the offset of the intersection
        """
        return self.kernel_api.get_intersection_offset(node_id)

    def change_intersection_offset(self, node_id, offset):
        """
        Changes an intersection's offset by the above offset

        Parameters
        ----------
        node_id : int
            the node id of the intersection
        offset : float
            the offset of the intersection

        Returns
        -------
        list
            list of current phases as ints
        """
        return self.kernel_api.change_intersection_offset(node_id, offset)

    
    def get_replication_name(self, node_id): #cj28
        """
        Gets the replication name

        Parameters
        ----------
        node_id : int
            the node id of the intersection

        Returns
        -------
        int
            the replication id/name
        """
        return self.kernel_api.get_replication_name(node_id)

    def get_cycle_length(self, node_id, control_id):  # cj
        """
        Gets the intersection's offset

        Parameters
        ----------
        node_id : int
            the node id of the intersection

        Returns
        -------
        int
            the offset of the intersection
        """
        return self.kernel_api.get_cycle_length(node_id, control_id)

    def get_duration_phase(self, node_id, phase):
        """
        Gets the intersection's offset

        Parameters
        ----------
        node_id : int
            the node id of the intersection
        phase: int
            phase index

        Returns
        -------
        int
            normal, max and min duration of phase
        """
        return self.kernel_api.get_duration_phase(node_id, phase)

    def get_control_ids(self, node_id):
        """
        Gets the intersection's offset

        Parameters
        ----------
        node_id : int
            the node id of the intersection

        Returns
        -------
        int
            the offset of the intersection
        """
        return self.kernel_api.get_control_ids(node_id)

    def get_green_phases(self, node_id, ring_id):
        """
        Gets the intersection's offset

        Parameters
        ----------
        node_id : int
            the node id of the intersection

        Returns
        -------
        int
            the offset of the intersection
        """
        return self.kernel_api.get_green_phases(node_id, ring_id)

    def change_phase_duration(self, node_id, phase, duration, maxout):
        """
        Changes an intersection's phase to the next

        Parameters
        ----------
        node_id : int
            the node id of the intersection
        duration: float
            the new phase duration of the control plan

        Returns
        -------
        int
            change phase duration
        """
        return self.kernel_api.change_phase_duration(node_id, phase, duration, maxout)

    def get_detector_lanes(self, edge_id):
        """
        Gets the detector ids on an edge

        Parameters
        ----------
        edge_id : int
            the id of the edge

        Returns
        -------
        list
            list of detector ids as ints
        """
        return self.kernel_api.get_detector_lanes(edge_id)

    def get_incoming_edges(self, node_id):
        """
        Gets an intersection's incoming edges

        Parameters
        ----------
        node_id : int
            the node id of the intersection

        Returns
        -------
        list
            list of edge ids as ints
        """
        return self.kernel_api.get_incoming_edges(node_id)

    def get_cumulative_queue_length(self, section_id):
        """
        Gets a section's cumulative queue length

        For some reason, AIMSUN only returns the cumulative length, not the actual values, hence,
        we do the calculations on the environment level.

        Parameters
        ----------
        section_id : int
            the id of the section

        Returns
        -------
        float
            the cumulative queue length
        """
        return self.kernel_api.get_cumulative_queue_length(section_id)

    def get_detectors_on_edge(self, edge_id):
        """
        Gets the detector ids on an edge

        Parameters
        ----------
        edge_id : int
            the id of the edge

        Returns
        -------
        list
            list of detector ids as ints
        """
        return self.kernel_api.get_detectors_on_edge(edge_id)

    def get_detector_count_and_occupancy(self, detector_id):
        """
        Gets the detector's flow and occupancy values

        Parameters
        ----------
        detector_id : int
            the id of the detector

        Returns
        -------
        int, float
            flow and occupancy of the detector
        """
        return self.kernel_api.get_detector_count_and_occupancy(detector_id)

    def set_replication_seed(self, seed):
        """
        Sets the replication seed

        Parameters
        ----------
        seed : int
            random seed
        """

        return self.kernel_api.set_replication_seed(seed)

    def get_green_util(self,node_id):
        """
        Gets the green time utilization at a specific node

        Parameters
        ----------
        node_id: int
            junction id
        """

        return self.kernel_api.get_green_util(node_id)

    def get_ave_app_delay(self,node_id):
        """
        Gets the green time utilization at a specific node

        Parameters
        ----------
        node_id: int
            junction id
        """

        return self.kernel_api.get_ave_app_delay(node_id)