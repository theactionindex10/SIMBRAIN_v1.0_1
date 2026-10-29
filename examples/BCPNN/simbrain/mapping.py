import os
from typing import Iterable, Optional, Union
from simbrain.memarray import MemristorArray
from simbrain.periphcircuit import DAC_Module
from simbrain.periphcircuit import ADC_Module
from simbrain.clipping import Clipping
import json
import pickle
import torch
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Mapping(torch.nn.Module):
    # language=rst
    """
    Abstract base class for mapping neural networks to memristor arrays.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the layer.
        :param memristor_info_dict: The parameters of the memristor device.
        :param CMOS_tech_info_dict: The parameters of CMOS technology.
        :param memristor_lut: The states of memristor under certain voltage.
        :param trans_ratio: Scale factor from conductance to x.
        :param learning: Whether to load with learning enabled. Default loads value from disk.
        """
        super().__init__()

        self.sim_params = sim_params
        self.device_name = sim_params['device_name']
        self.device_structure = sim_params['device_structure']
        self.CMOS_technode = sim_params['CMOS_technode']
        self.device_roadmap = sim_params['device_roadmap']
        self.input_bit = sim_params['input_bit']
        self.ADC_setting = sim_params['ADC_setting']
        self.ADC_rounding_function = sim_params['ADC_rounding_function']

        if self.device_structure == 'STDP_crossbar':
            self.shape = [1, 1]  # Shape of the memristor crossbar
            for element in shape:
                self.shape[0] *= element #Row-wise update
                self.shape[1] *= element #Column-wise update
            self.shape[0] = int(self.shape[0] ** (1/2))
            self.shape[1] = int(self.shape[1] ** (1/2))
            self.shape = tuple(self.shape)
        elif self.device_structure == 'trace':
            self.shape = [1, 1]  # Shape of the memristor crossbar
            for element in shape:
                self.shape[1] *= element #single row
            self.shape = tuple(self.shape)
        elif self.device_structure == 'crossbar':
            self.shape = shape
        else:
            raise Exception("Only trace and crossbar architecture are supported!")
        
        self.register_buffer("mem_v", torch.Tensor())
        self.register_buffer("mem_x_read", torch.Tensor())
        self.register_buffer("mem_t", torch.Tensor())
        self.register_buffer("mem_wr_t", torch.Tensor())

        with open('../../memristor_device_info.json', 'r') as f:
            self.memristor_info_dict = json.load(f)
        assert self.device_name in self.memristor_info_dict.keys(), "Invalid Memristor Device!"
        self.Gon = self.memristor_info_dict[self.device_name]['G_on']
        self.Goff = self.memristor_info_dict[self.device_name]['G_off']
        self.v_read = self.memristor_info_dict[self.device_name]['v_read']

        with open('../../CMOS_tech_info.json', 'r') as f:
            self.CMOS_tech_info_dict = json.load(f)
        assert self.device_roadmap in self.CMOS_tech_info_dict.keys(), "Invalid Memristor Device!"
        assert str(self.CMOS_technode) in self.CMOS_tech_info_dict[self.device_roadmap].keys(), "Invalid Memristor Device!"

        with open('../../memristor_lut.pkl', 'rb') as f:
            self.memristor_luts = pickle.load(f)
        assert self.device_name in self.memristor_luts.keys(), "No Look-Up-Table Data Available for the Target Memristor Type!"

        self.trans_ratio = 1 / (self.Goff - self.Gon)

        self.batch_size = None
        self.learning = None

        self.sim_power = {}
        self.sim_periph_power = {}
        self.sim_area = {}


    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when memristor is used to mapping NN applications.
    
        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        self.mem_v = torch.zeros(batch_size, *self.shape, device=self.mem_v.device)
        self.mem_x_read = torch.zeros(batch_size, 1, self.shape[1], device=self.mem_x_read.device)
        self.mem_t = torch.zeros(batch_size, *self.shape, device=self.mem_t.device, dtype=torch.int64)
        self.mem_wr_t = torch.zeros(batch_size, *self.shape, device=self.mem_wr_t.device, dtype=torch.int)       

'''IMPORTANT FOR BCPNN: Will need to create a new one  '''
class STDPMapping(Mapping):
    # language=rst
    """
    Mapping STDP (Bindsnet) to memristor arrays.
    """
    
    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the memristor array.
        :param vneg: Negative voltage applied to the memristor.
        :param vpos: Positive voltage applied to the memristor.
        :param trace_decay: The original trace drop per time step.
        """
        super().__init__(
            sim_params=sim_params,
            shape=shape
        )
	#Submodules
        self.mem_array = MemristorArray(sim_params=sim_params, shape=self.shape,
                                        memristor_info_dict=self.memristor_info_dict)
        self.DAC_module = DAC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
        self.ADC_module = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
        if 'clipping' in sim_params.keys():
            self.clipping = Clipping(sim_params=sim_params, shape=self.shape, memristor_info_dict=self.memristor_info_dict)

        if self.device_structure == 'STDP_crossbar':
            self.batch_interval = sim_params['batch_interval'] * self.shape[0] * 3 + 1
            self.write_batch_interval = sim_params['batch_interval'] + 1
        elif self.device_structure == 'trace':
            self.batch_interval = sim_params['batch_interval'] * 2 + 1
            self.write_batch_interval = sim_params['batch_interval'] + 1

        self.register_buffer("mem_v_read", torch.Tensor())
        self.register_buffer("x", torch.Tensor())
        self.register_buffer("s", torch.Tensor())
        self.vneg = 0
        self.vpos = 0
        self.trace_decay = 0


    def set_batch_size_stdp(self, batch_size, learning) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when memristor is used to mapping trace-STDP.
    
        :param batch_size: Mini-batch size.
        :param learning: Whether to load with learning enabled. Default loads value from disk.
        """
        self.learning = learning
        self.set_batch_size(batch_size)
        self.mem_array.set_batch_size(batch_size=self.batch_size)
        self.DAC_module.set_batch_size(batch_size=batch_size)
        self.ADC_module.set_batch_size(batch_size=batch_size)

        self.mem_v_read = torch.zeros(1, batch_size, 1, self.shape[0], device=self.mem_v_read.device)
        self.x = torch.zeros(batch_size, *self.shape, device=self.x.device)
        self.s = torch.zeros(batch_size, *self.shape, device=self.s.device)

        if self.learning:
            mem_t_matrix = (self.batch_interval * torch.arange(self.batch_size, device=self.mem_t.device))
            self.mem_t[:, :, :] = mem_t_matrix.view(-1, 1, 1)
            mem_wr_t_matrix = (self.write_batch_interval * torch.arange(self.batch_size, device=self.mem_t.device))
            self.mem_wr_t[:, :, :] = mem_wr_t_matrix.view(-1, 1, 1)            
        else:
            self.mem_t.fill_(torch.min(self.mem_t_batch_update[:]))
            self.mem_wr_t.fill_(torch.min(self.mem_wr_t_batch_update[:]))
            
        self.mem_array.mem_t = self.mem_t
        self.mem_array.mem_wr_t = self.mem_wr_t
        
        
        


    '''IMPORTANT: WILL NEED THIS TO CHANGE FOR BCPNN'''
    def voltage_generation(self, trace_decay, plot) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when memristor is used to mapping trace-STDP.
    
        :param trace_decay: The original trace drop per time step.
        :param plot: A boolean flag to determine whether to plot the results.
        """
        # Simulation Setup
        points = 150
        spike = torch.zeros(points)
        ori_trace = torch.zeros(points)
        mem_x = torch.zeros(points)

        # STDP Setup
        spike[10] = 1

        # Original Trace - waits until spike[10], jumps up, and then clamps the value to 1 if the condition is met.
        for i in range(len(spike) - 1):
            ori_trace[i + 1] = ori_trace[i] * trace_decay + spike[i]

            if ori_trace[i + 1] > 1:
                ori_trace[i + 1] = 1
			
        # Memristor-based Trace
        # Do not count in non-idealities
        # Baseline model
        test_sim_params = {'device_structure': self.sim_params['device_structure'],
                           'device_name': self.sim_params['device_name'],
                           'c2c_variation': False,
                           'd2d_variation': 0,
                           'stuck_at_fault': False,
                           'retention_loss': 0,
                           'aging_effect': 0,
                           'wire_width': self.sim_params['wire_width'],
                           'input_bit': self.sim_params['input_bit'],
                           'batch_interval': self.sim_params['batch_interval'],
                           'CMOS_technode': self.sim_params['CMOS_technode'],
                           'ADC_precision': self.sim_params['ADC_precision'],
                           'ADC_setting': self.sim_params['ADC_setting'],
                           'ADC_rounding_function': self.sim_params['ADC_rounding_function'],
                           'device_roadmap': self.sim_params['device_roadmap'],
                           'temperature': self.sim_params['temperature'],
                           'hardware_estimation': self.sim_params['hardware_estimation']}
        test_array = MemristorArray(sim_params=test_sim_params, shape=(1, 1), memristor_info_dict=self.memristor_info_dict)
        test_array.set_batch_size(batch_size=1)
        mem_info = self.memristor_info_dict[self.device_name]

        dt = mem_info['delta_t'] * mem_info['duty_ratio']
        k_off = mem_info['k_off']
        v_off = mem_info['v_off']
        alpha_off = mem_info['alpha_off']
        v_pos = v_off * (math.pow(1 / (dt * k_off), 1.0 / alpha_off) + 1)

        if self.device_structure == 'STDP_crossbar':
            write_time = 2
        elif self.device_structure == 'trace':
            write_time = 1

        if mem_info['P_on'] == 1:
            k_on = mem_info['k_on']
            v_on = mem_info['v_on']
            alpha_on = mem_info['alpha_on']
            v_neg = v_on * (math.pow((trace_decay - 1) / (dt * k_on), 1.0 / alpha_on) + 1)
        else:
            # Enable batch processing for searching the best v_neg
            v_on = mem_info['v_on']
            n_test = 500
            v_tensor = torch.arange(v_on, v_on - 0.01 * n_test, -0.01)

            test_array.set_batch_size(batch_size=n_test)
            test_x = torch.zeros(points, n_test, 1, 1)

            for t in range(points - 1):
                mem_s = torch.tensor(spike[t], dtype=torch.float64)
                mem_v = v_tensor if mem_s == 0 else torch.tensor(v_pos).expand(n_test)

                mem_c = test_array.memristor_write(mem_v=mem_v.unsqueeze(1).unsqueeze(2), write_time=write_time, mem_v_amp=[0,0])
                test_x[t + 1] = (mem_c - self.Gon) * self.trans_ratio

            # Compare results
            golden_x = ori_trace.reshape(points, 1, 1, 1).expand(-1, n_test, -1, -1)
            mse = torch.zeros(n_test)
            for i in range(n_test):
                mse[i] = F.mse_loss(test_x[:, i, :, :], golden_x[:, i, :, :])
            min_mse, min_index = torch.min(mse, 0)
            v_neg = float(v_tensor[min_index])

        if plot:
            blue = (47 / 255, 130 / 255, 189 / 255)
            green = (98 / 255, 149 / 255, 61 / 255)

            plt.figure(figsize=(13, 4.5))
            grid = plt.GridSpec(14, 17, wspace=0.5, hspace=0.5)
            ax = plt.subplot(grid[0:14, 0:17])

            test_array.set_batch_size(batch_size=1)
            for t in range(points - 1):
                mem_s = torch.tensor(spike[t], dtype=torch.float64)
                mem_v = mem_s.unsqueeze(0).unsqueeze(0).unsqueeze(1)
                mem_v[mem_v == 0] = v_neg
                mem_v[mem_v == 1] = v_pos

                mem_c = test_array.memristor_write(mem_v=mem_v, write_time=write_time, mem_v_amp=[0,0])

                # mem to nn
                temp_x = (mem_c - self.Gon) * self.trans_ratio
                mem_x[t+1] = temp_x.squeeze()

            # Plot the original trace and memristor trace
            plot_x = range(points)
            # Original
            ax.plot(ori_trace, color=blue, label='Original Trace')
            ax.plot(mem_x, color=green, label='Memristor Trace')
            ax.legend(frameon=False)

            plt.tight_layout()
            plt.savefig('voltage_generation.png', dpi=300, bbox_inches='tight')
            plt.show()

        self.vneg = v_neg
        self.vpos = v_pos


    def mapping_write_stdp(self, s):
        # language=rst
        """
        simulates the process of mapping NN to the memristor array for both STDP_crossbar and trace structures.
    
        :param s: Input spikes.
        :param x: Internal states of memristors.
        """
        write_time = 1
        if s.dim() == 4:
            self.s = s.flatten(2, 3)
        elif s.dim() == 2:
            self.s = torch.unsqueeze(s, 1)
        
        # nn to mem
        self.mem_v = self.s.float()
        self.mem_v[self.mem_v == 0] = self.vneg
        self.mem_v[self.mem_v == 1] = self.vpos      

        self.DAC_module.DAC_write(mem_v=self.mem_v, mem_v_amp=[self.vpos, self.vneg])

        mem_c = self.mem_array.memristor_write(mem_v=self.mem_v, write_time=write_time, mem_v_amp=[self.vpos, self.vneg])
        
        # mem to nn
        self.x = (mem_c - self.Gon) * self.trans_ratio

        if self.device_structure == 'STDP_crossbar':
            if s.dim() == 4:
                self.x = self.x.unsqueeze(1)
            elif s.dim() == 2:
                self.x = self.x.flatten(1, 2)
        elif self.device_structure == 'trace':
            if s.dim() == 4:
                self.x = self.x.reshape(s.size(0), s.size(1), s.size(2), s.size(3))
            elif s.dim() == 2:
                self.x = self.x.squeeze()

        return self.x


    def mapping_read_stdp(self, s):
        # language=rst
        """
   	 simulates the process of reading traces for trace (BCPNN cannot be handled by STDP crossbar
        :param s: Input spikes.
        :param mem_x_read: Internal states of memristors.
        """
        if s.dim() == 4:
            s = s.flatten(2, 3)
        elif s.dim() == 2:
            s = torch.unsqueeze(s, 1)

            # Read Voltage generation
            # For every batch, read is not necessary when there is no spike s
        s_sum = torch.sum(s, dim=2).squeeze()
        s_sum = torch.unsqueeze(s_sum, 1)

        self.mem_v_read.zero_()
        self.mem_v_read[0, s_sum.bool()] = 1

        self.mem_v_read = self.DAC_module.DAC_read(mem_v=self.mem_v_read, sgn=None)

        mem_i, _ = self.mem_array.memristor_read(mem_v=self.mem_v_read, read_time=1)
        ADC_mem_c = 1 / (1 / self.Goff + self.mem_array.total_wire_resistance)
        mem_i = self.ADC_module.ADC_read(mem_i_sequence=mem_i, mem_c=ADC_mem_c, high_cut_ratio=1)

        if 'clipping' in self.sim_params.keys():
            mem_i = self.clipping.clipping_function(mem_i_origin=mem_i)

        # current to trace
        self.mem_x_read = (mem_i/self.v_read - self.Gon) * self.trans_ratio
        self.mem_x_read[~s_sum.bool()] = 0

        return self.mem_x_read


    def reset_memristor_variables(self) -> None:
        # language=rst
        """
        Abstract base class method for resetting state variables.
        """
        v_reset = self.memristor_luts[self.device_name]['V_reset']
        self.mem_v.fill_(v_reset)       
        # Adopt large negative pulses to reset the memristor array
        self.DAC_module.DAC_reset(mem_v=self.mem_v)
        self.mem_array.memristor_reset(mem_v=self.mem_v)


    def mem_t_update(self) -> None:
        # language=rst
        """
        Updates the timing parameters for the memristor array.
        """
        self.mem_array.mem_t += self.batch_interval * (self.batch_size - 1)
        self.mem_array.mem_wr_t += self.write_batch_interval * (self.batch_size - 1)

    def update_SAF_mask(self) -> None:
        # language=rst
        """
        Updates the Stuck-At Fault (SAF) mask for the memristor array.
        """
        self.mem_array.update_SAF_mask()

    def total_area_calculation(self) -> None:
        # language=rst
        """
        Calculate total area for memristor-based architecture. Called when power is reported.
        """
        self.sim_mem_area = self.mem_array.area.array_area

        DAC_height_row, DAC_width_row, DAC_height_col, DAC_width_col, sim_switch_matrix_row_area, sim_switch_matrix_col_area = self.DAC_module.DAC_module_area.DAC_module_cal_area()
        ADC_height, ADC_width, sim_shiftadd_area, sim_SarADC_area = self.ADC_module.ADC_module_area.ADC_module_cal_area()
        periph_total_area = sim_switch_matrix_row_area + sim_switch_matrix_col_area + sim_shiftadd_area + sim_SarADC_area
        self.sim_periph_area = {'sim_switch_matrix_row_area': sim_switch_matrix_row_area,
                             'sim_switch_matrix_col_area': sim_switch_matrix_col_area,
                             'sim_shiftadd_area': sim_shiftadd_area, 'sim_SarADC_area': sim_SarADC_area,
                             'sim_total_periph_area': periph_total_area}

        total_height = max(self.mem_array.length_col + ADC_height + DAC_height_col, DAC_height_row)
        total_width = DAC_width_row + max(self.mem_array.length_row, DAC_width_col, ADC_width)
        self.sim_total_area = total_height * total_width

        self.sim_area = {'sim_mem_area':self.sim_mem_area,
                         'sim_periph_area':periph_total_area,
                         'sim_total_area':self.sim_total_area,
                         'sim_used_area_ratio':(self.sim_mem_area+periph_total_area)/self.sim_total_area}

       
class BCPNNMapping(Mapping):
    # language=rst
    """
    Mapping BCPNN to memristor arrays.
    """
    
    def __init__(
        self,
        sim_params: dict = {},
        shape: dict = {'Pi' : 47040, 'Pj' : 3000, 'Pji' : 4704000},
        traces: Iterable[str] = ('Zi', 'Zj', 'Pi', 'Pj'),
        last_read: dict = {},
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the memristor array.
        :param vneg: Negative voltage applied to the memristor.
        :param vpos: Positive voltage applied to the memristor.
        """
        super().__init__(
            sim_params=sim_params,
            shape=(max(shape.values()),)
        )

        # One memristor array per trace, keyed by the same names as v_pair
        self.mem_arrays = torch.nn.ModuleDict()
        for name, size in shape.items():
            self.mem_arrays[name] = MemristorArray(sim_params=sim_params, shape=(1,size),
                                                   memristor_info_dict=self.memristor_info_dict)
        self.DAC_module = DAC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
        self.ADC_module = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
        if 'clipping' in sim_params.keys():
            self.clipping = Clipping(sim_params=sim_params, shape=self.shape, memristor_info_dict=self.memristor_info_dict)

        if self.device_structure == 'STDP_crossbar':
            self.batch_interval = sim_params['batch_interval'] * self.shape[0] * 3 + 1
            self.write_batch_interval = sim_params['batch_interval'] + 1
        elif self.device_structure == 'trace':
            self.batch_interval = sim_params['batch_interval'] * 2 + 1
            self.write_batch_interval = sim_params['batch_interval'] + 1

        self.register_buffer("mem_v_read", torch.Tensor())
        self.register_buffer("x", torch.Tensor())
        self.register_buffer("s", torch.Tensor())
        self.vneg = 0
        self.vpos = 0
        self.v_pair = {}  # trace name -> (v_pos, v_neg), filled by voltage_generation

    def set_batch_size_bcpnn(self, batch_size, learning) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when memristor is used to mapping trace-STDP.
    
        :param batch_size: Mini-batch size.
        :param learning: Whether to load with learning enabled. Default loads value from disk.
        """
        self.learning = learning
        self.set_batch_size(batch_size)
        for mem_array in self.mem_arrays.values():
            mem_array.set_batch_size(batch_size=self.batch_size)
        self.DAC_module.set_batch_size(batch_size=batch_size)
        self.ADC_module.set_batch_size(batch_size=batch_size)

        self.mem_v_read = torch.zeros(1, batch_size, 1, self.shape[0], device=self.mem_v_read.device)
        self.x = torch.zeros(batch_size, *self.shape, device=self.x.device)
        self.s = torch.zeros(batch_size, *self.shape, device=self.s.device)
        

        if self.learning:
            mem_t_matrix = (self.batch_interval * torch.arange(self.batch_size, device=self.mem_t.device))
            self.mem_t[:, :, :] = mem_t_matrix.view(-1, 1, 1)
            mem_wr_t_matrix = (self.write_batch_interval * torch.arange(self.batch_size, device=self.mem_t.device))
            self.mem_wr_t[:, :, :] = mem_wr_t_matrix.view(-1, 1, 1)            
        else:
            self.mem_t.fill_(torch.min(self.mem_t_batch_update[:]))
            self.mem_wr_t.fill_(torch.min(self.mem_wr_t_batch_update[:]))
            
        for mem_array in self.mem_arrays.values():
            mem_array.mem_t = self.mem_t.clone()
            mem_array.mem_wr_t = self.mem_wr_t.clone()
            
        self.last_read = {}

        
        
        
    def voltage_generation_bcpnn(self, trace_file, taupdt=0.001, plot=False) -> None:
        # language=rst
        """
        Calibrates one (v_pos, v_neg) pair per trace and stores them in ``self.v_pair`` by name.
        Called from Nodes.compute_decays, which supplies the per-step constants (Wang et al. 2021, Eq. 9).

        :param kzi: Z_i update constant, kzi = dt / tau_zi.
        :param kzj: Z_j update constant, kzj = dt / tau_zj.
        :param kp: P update constant, kp = dt / tau_p.
        :param plot: A boolean flag to determine whether to plot the results.
        :param calib_rate: Firing probability per step of the calibration spike trains. It should resemble the
            rate the layer sees in operation (Poisson-encoded MNIST at intensity 64: about 0.06 for a bright pixel),
            because the golden P that the P pairs are fitted to is built from these trains.
        :param calib_points: Length of the calibration run in steps. Must be long against tau_p for P to develop.
        :param calib_seed: Seed for the calibration spike trains, so calibration is reproducible.
        """
        
        #Taken from BCPNNSim2.1 code - trace values calculated from Prj::updtraces
        cache_file = os.path.splitext(trace_file)[0] + '_calib.npz'
        if os.path.exists(cache_file) and os.path.getmtime(cache_file) >= os.path.getmtime(trace_file):
            c = np.load(cache_file, allow_pickle=True)
            self.v_pair = c['v_pair'].item()
            self.group = {name: torch.from_numpy(c['group_' + name]) for name in self.v_pair}
            self._set_cell_voltages()
            return
        d = np.load(trace_file)
        axoact, hihjhi, zj = d['axoact'], d['hihjhi'], d['zj']
        pi, pj, pji, pji_index = d['pi'], d['pj'], d['pji'], d['pji_index']
        T = axoact.shape[0]
        Hj, denHi, Mi = 30, 784, 2
        denNi, Mj = denHi * Mi, 100
        eps = 1e-8
        
        #rebuilt Zi from AXOACT through the hypercolumn, minicolumn slot table 
        hih = hihjhi.reshape(T, Hj, denHi).astype(int)
        zi = np.empty((T,Hj*denNi))
        for t in range(T):
            src = (hih[t][:, :, None] * Mi + np.arange(Mi)[None, None, :]).reshape(Hj, denNi)
            zi[t] = (axoact[t][src] * (1-eps) + eps).reshape(-1)
        
        # Pji final values for ALL synapses, integrated from the drives (only 1000 columns of Pji are recorded,
        # but every synapse's drive Zi*Zj is).  Layout nj*denNi + dni, as the engine.  Checked against the
        # recorded columns below.
        pji_final = np.zeros(Hj * Mj * denNi)
        for t in range(T):
            drive_pji = (zi[t].reshape(Hj, 1, denNi) * zj[t].reshape(Hj, Mj, 1)).reshape(-1)
            pji_final += (drive_pji - pji_final) * taupdt
        assert np.abs(pji_final[pji_index] - pji[-1]).max() < 1e-6, 'Pji integration vs recording'

        # Group the cells of every trace into three bands by their final golden value: quiet, middle, active.
        # The band edges are placed on the VALUE axis, log-spaced between a floor and the maximum, so every band
        # spans the same factor (cells below the floor fall into the quiet band). One floor ratio for all arrays.
        # Each band gets its own representative, the cell nearest the median
        # of the band's cells AT OR ABOVE THE FLOOR (so the quiet pair is fitted to a driven cell, not to the
        # tail), and its own voltage pair.
        floor_ratio = 1e-2
        finals = {'Pi': pi[-1], 'Pj': pj[-1], 'Pji': pji_final}
        self.group = {}
        reps = {}
        for name in ('Pi', 'Pj', 'Pji'):
            final = finals[name]
            top = final.max()
            floor = max(final.min(), top * floor_ratio)
            bounds = np.logspace(np.log10(floor), np.log10(top), 4)        # 4 edges -> 3 bands, equal factor
            band = np.digitize(final, bounds[1:-1])                         # 0 = quiet, 1 = middle, 2 = active
            i_q = np.where(band == 0)[0]
            i_m = np.where(band == 1)[0]
            i_a = np.where(band == 2)[0]
            quiet = final[i_q]
            middle = final[i_m]
            active = final[i_a]
            q_index = int(np.argmin(np.abs(quiet - np.median(quiet[quiet >= floor]))))
            m_index = int(np.argmin(np.abs(middle - np.median(middle[middle >= floor]))))
            a_index = int(np.argmin(np.abs(active - np.median(active[active >= floor]))))
            reps[name] = {'quiet': i_q[q_index], 'middle': i_m[m_index], 'active': i_a[a_index]}
            # Label of every cell (used by mapping_write_bcpnn)
            self.group[name] = torch.from_numpy(band.astype(np.int64)).reshape(1, -1)

        drives = {'Pi': {}, 'Pj': {}, 'Pji': {}}
        recorded = {'Pi': {}, 'Pj': {}, 'Pji': {}}
        for gname in reps['Pi']:                              # band names in label order
            drives['Pi'][gname] = zi[:, reps['Pi'][gname]]
            recorded['Pi'][gname] = pi[:, reps['Pi'][gname]]
            drives['Pj'][gname] = zj[:, reps['Pj'][gname]]
            recorded['Pj'][gname] = pj[:, reps['Pj'][gname]]
            k = reps['Pji'][gname]; nj = k // denNi; dni = k % denNi; hj = nj // Mj
            drives['Pji'][gname] = zi[:, hj * denNi + dni] * zj[:, nj]
            # the engine's trace of this synapse is recorded only if it is one of the 1000 columns
            hit = np.where(pji_index == k)[0]
            recorded['Pji'][gname] = pji[:, hit[0]] if len(hit) else None
        
        
        
        # Memristor-based Trace
        # Do not count in non-idealities
        test_sim_params = {'device_structure': self.sim_params['device_structure'],
                           'device_name': self.sim_params['device_name'],
                           'c2c_variation': False,
                           'd2d_variation': 0,
                           'stuck_at_fault': False,
                           'retention_loss': 0,
                           'aging_effect': 0,
                           'wire_width': self.sim_params['wire_width'],
                           'input_bit': self.sim_params['input_bit'],
                           'batch_interval': self.sim_params['batch_interval'],
                           'CMOS_technode': self.sim_params['CMOS_technode'],
                           'ADC_precision': self.sim_params['ADC_precision'],
                           'ADC_setting': self.sim_params['ADC_setting'],
                           'ADC_rounding_function': self.sim_params['ADC_rounding_function'],
                           'device_roadmap': self.sim_params['device_roadmap'],
                           'temperature': self.sim_params['temperature'],
                           'hardware_estimation': self.sim_params['hardware_estimation']}
        test_array = MemristorArray(sim_params=test_sim_params, shape=(1, 1), memristor_info_dict=self.memristor_info_dict)
        test_array.set_batch_size(batch_size=1)
        self.v_pair = {}       # name -> {group name: (v_pos, v_neg)}, groups in label order 0, 1, 2
        for name in ('Pi', 'Pj', 'Pji'):
            self.v_pair[name] = {}
            for gname in drives[name]:
                drive = torch.tensor(np.concatenate([drives[name][gname][:1], drives[name][gname]]), dtype=torch.float64)
                golden = torch.zeros(T + 1, dtype=torch.float64)
                for t in range(1, T + 1):                               # Prj::updtraces: same-step drive
                    golden[t] = golden[t - 1] + (drive[t] - golden[t - 1]) * taupdt
                if recorded[name][gname] is not None:                  # transcription check where the engine trace exists
                    assert np.abs(golden[1:].numpy() - recorded[name][gname]).max() < 1e-6, (name, gname)
                self.v_pair[name][gname] = self._calibrate_driven(drive, golden, test_array, plot, name=f'{name}_{gname}')
        self._set_cell_voltages()
        np.savez(cache_file, v_pair=np.array(self.v_pair, dtype=object),
                 **{'group_' + name: self.group[name].numpy() for name in self.v_pair})

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        dev = self.mem_v.device
        for d in ('vpos_cell', 'vneg_cell', 'group'):
            if hasattr(self, d):
                setattr(self, d, {k: v.to(dev) for k, v in getattr(self, d).items()})
        self.last_read = {}
        return self

    def _set_cell_voltages(self):
        self.vpos_cell, self.vneg_cell = {}, {}
        for name in self.v_pair:
            pairs = list(self.v_pair[name].values())
            vpos_tab = torch.tensor([q[0] for q in pairs], dtype=torch.float32)
            vneg_tab = torch.tensor([q[1] for q in pairs], dtype=torch.float32)
            self.vpos_cell[name] = vpos_tab[self.group[name]].unsqueeze(1)
            self.vneg_cell[name] = vneg_tab[self.group[name]].unsqueeze(1)
        
  #Comparing the memristor-based trace to the golden traces. The golden traces are obtained from BCPNNSim2.1
    def _calibrate_driven(self, drive, trace, test_array, plot=False, name=''):
        # language=rst
        
        #drive = median cell's drive, trace = gated values 
        points = len(trace)
        mem_info = self.memristor_info_dict[self.device_name]
        write_time = 1

        #Voltage pair candidates 
        v_off = mem_info['v_off']
        v_on = mem_info['v_on']
        n_pos = 400
        n_neg = 400
        v_pos_tensor = v_off * (1 + torch.logspace(-3, math.log10(1.5), n_pos)) #Allows for finer grid, more voltage candidates
        v_neg_tensor = v_on * (1 + torch.logspace(-3, math.log10(1.5), n_neg))

        # Enable batch processing for searching the best (v_pos, v_neg) pair: one memristor per combination.
        
        n_test = n_pos * n_neg
        v_pos_grid = v_pos_tensor.repeat_interleave(n_neg)
        v_neg_grid = v_neg_tensor.repeat(n_pos)

        test_array.set_batch_size(batch_size=n_test)
        test_x = torch.zeros(points, n_test, 1, 1)

        for t in range(points - 1):
            
            mem_d = torch.tensor(drive[t+1], dtype=torch.float64)
            mem_up = mem_d > test_x[t, :, 0, 0]
            mem_v = torch.zeros(n_test)
            mem_v[mem_up] = v_pos_grid[mem_up]
            mem_v[~mem_up] = v_neg_grid[~mem_up]

            mem_c = test_array.memristor_write(mem_v=mem_v.unsqueeze(1).unsqueeze(2), write_time=write_time, mem_v_amp=[0,0])
            test_x[t + 1] = (mem_c - self.Gon) * self.trans_ratio

        # Compare results
        golden_x = trace.reshape(points, 1, 1, 1).expand(-1, n_test, -1, -1)
        mse = torch.zeros(n_test)
        for i in range(n_test):
            mse[i] = F.mse_loss(test_x[:, i, :, :], golden_x[:, i, :, :])
        min_mse, min_index = torch.min(mse, 0)
        v_pos = float(v_pos_grid[min_index])
        v_neg = float(v_neg_grid[min_index])

        if plot:
            blue = (47 / 255, 130 / 255, 189 / 255)
            green = (98 / 255, 149 / 255, 61 / 255)
            mem_x = torch.zeros(points)

            plt.figure(figsize=(13, 4.5))
            grid = plt.GridSpec(14, 17, wspace=0.5, hspace=0.5)
            ax = plt.subplot(grid[0:14, 0:17])

            test_array.set_batch_size(batch_size=1)
            for t in range(points - 1):
                mem_d = torch.tensor(drive[t+1], dtype=torch.float64)
                mem_v = torch.zeros(1, 1, 1)
                mem_v[:] = v_pos if mem_d > mem_x[t] else v_neg

                mem_c = test_array.memristor_write(mem_v=mem_v, write_time=write_time, mem_v_amp=[0,0])

                # mem to nn
                temp_x = (mem_c - self.Gon) * self.trans_ratio
                mem_x[t+1] = temp_x.squeeze()

            # Plot the original trace and memristor trace
            ax.plot(trace, color=blue, label='Original ' + name)
            ax.plot(mem_x, color=green, label='Memristor ' + name)
            ax.legend(frameon=False)

            plt.tight_layout()
            plt.savefig(f'voltage_generation_{name}.png', dpi=300, bbox_inches='tight')
            plt.show()

        return v_pos, v_neg
        
    def mapping_write_bcpnn(self, drive, x_current=None, trace=None):
        # language=rst
        """
        simulates the process of mapping BCPNN to the memristor array for trace structure.
    
        :param drive: Input spikes.
        :param x_current: No longer used for the up/down decision (kept so existing callers still run);
                          the state is read from the array through the ADC (mapping_read_bcpnn), as the
                          controller would in hardware.  Decided 2026-10-27.
        :param trace: Current trace being written 
        """
        write_time = 1
        #if drive.dim() == 4:
        #    self.s = drive.flatten(2, 3)
        #    x_cur = x_current.flatten(2,3)
        
        self.s = torch.unsqueeze(drive, 1)                 # (1 ,N) -> (1, 1, N) 
        if trace not in self.last_read:
            self.last_read[trace] = self.mapping_read_bcpnn(trace)            
        x_cur = self.last_read[trace]
        
        #elif drive.dim() == 3:
        #    self.s = drive.flatten(1, 2).unsqueeze(1)          # (B, H, M) -> (B, 1, H*M) single memristor row
        #    x_cur = x_current.flatten(1, 2).unsqueeze(1)	
        
        # nn to mem
        mem_up = self.s > x_cur
        self.mem_v = torch.where(mem_up, self.vpos_cell[trace], self.vneg_cell[trace])
        v_amp = [float(self.vpos_cell[trace].max()), float(self.vneg_cell[trace].min())]

        self.DAC_module.DAC_write(mem_v=self.mem_v, mem_v_amp=v_amp)

        mem_c = self.mem_arrays[trace].memristor_write(mem_v=self.mem_v, write_time=write_time, mem_v_amp=v_amp)
        
        # mem to nn
        self.x = (mem_c - self.Gon) * self.trans_ratio

        self.last_read[trace] = self.mapping_read_bcpnn(trace)
        self.x = self.last_read[trace].reshape(drive.shape)
        
        return self.x
        
    def mapping_read_bcpnn(self, trace):
        # language=rst
        """
   	 simulates the process of reading traces for trace (BCPNN cannot be handled by STDP crossbar)
        :param s: Input spikes.
        :param mem_x_read: Internal states of memristors.
        """
        self.mem_v_read.fill_(1)

        # Read Voltage generation
     

        self.mem_v_read = self.DAC_module.DAC_read(mem_v=self.mem_v_read, sgn=None)

        mem_i, _ = self.mem_arrays[trace].memristor_read(mem_v=self.mem_v_read, read_time=1)
        ADC_mem_c = 1 / (1 / self.Goff + self.mem_arrays[trace].total_wire_resistance)
        mem_i = self.ADC_module.ADC_read(mem_i_sequence=mem_i, mem_c=ADC_mem_c, high_cut_ratio=1)

        if 'clipping' in self.sim_params.keys():
            mem_i = self.clipping.clipping_function(mem_i_origin=mem_i)

        # current to trace
        self.mem_x_read = (mem_i/self.v_read - self.Gon) * self.trans_ratio
        

        return self.mem_x_read               
        
  
    def reset_memristor_variables(self) -> None:
        # language=rst
        """
        Abstract base class method for resetting state variables per trace.
        """
        v_reset = self.memristor_luts[self.device_name]['V_reset']
        self.mem_v.fill_(v_reset)
        # Adopt large negative pulses to reset the memristor array
        self.DAC_module.DAC_reset(mem_v=self.mem_v)
        for mem_array in self.mem_arrays.values():
            mem_array.memristor_reset(mem_v=self.mem_v)

    def mem_t_update(self) -> None:
        # language=rst
        """
        Updates the timing parameters for the memristor arrays per trace.
        """
        for mem_array in self.mem_arrays.values():
            mem_array.mem_t += self.batch_interval * (self.batch_size - 1)
            mem_array.mem_wr_t += self.write_batch_interval * (self.batch_size - 1)

    def update_SAF_mask(self) -> None:
        # language=rst
        """
        Updates the Stuck-At Fault (SAF) mask for the memristor array.
        """
        for mem_array in self.mem_arrays.values():
            mem_array.update_SAF_mask()

    def total_area_calculation(self) -> None:
        # language=rst
        """
        Calculate total area for memristor-based architecture. Called when power is reported.
        """
        self.sim_mem_area = sum(a.area.array_area for a in self.mem_arrays.values())
        
        any_array = next(iter(self.mem_arrays.values()))

        DAC_height_row, DAC_width_row, DAC_height_col, DAC_width_col, sim_switch_matrix_row_area, sim_switch_matrix_col_area = self.DAC_module.DAC_module_area.DAC_module_cal_area()
        ADC_height, ADC_width, sim_shiftadd_area, sim_SarADC_area = self.ADC_module.ADC_module_area.ADC_module_cal_area()
        periph_total_area = sim_switch_matrix_row_area + sim_switch_matrix_col_area + sim_shiftadd_area + sim_SarADC_area
        self.sim_periph_area = {'sim_switch_matrix_row_area': sim_switch_matrix_row_area,
                             'sim_switch_matrix_col_area': sim_switch_matrix_col_area,
                             'sim_shiftadd_area': sim_shiftadd_area, 'sim_SarADC_area': sim_SarADC_area,
                             'sim_total_periph_area': periph_total_area}

        total_height = max(any_array.length_col + ADC_height + DAC_height_col, DAC_height_row)
        total_width = DAC_width_row + max(sum(a.length_row for a in self.mem_arrays.values()), DAC_width_col, ADC_width)
        self.sim_total_area = total_height * total_width

        self.sim_area = {'sim_mem_area':self.sim_mem_area,
                         'sim_periph_area':periph_total_area,
                         'sim_total_area':self.sim_total_area,
                         'sim_used_area_ratio':(self.sim_mem_area+periph_total_area)/self.sim_total_area}


class MLPMapping(Mapping):
    # language=rst
    """
    Mapping Multi-Layer Perceptron (MLP) to memristor arrays.
    """

    def __init__(
            self,
            sim_params: dict = {},
            shape: Optional[Iterable[int]] = None,
            **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the memristor array.
        """
        super().__init__(
            sim_params=sim_params,
            shape=shape
        )

        self.register_buffer("norm_ratio", torch.Tensor())
        self.register_buffer("write_pulse_no", torch.Tensor())

        # Corssbar for positive input and positive weight
        self.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=self.shape,
                                        memristor_info_dict=self.memristor_info_dict)
        # Corssbar for negative input and positive weight
        self.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=self.shape,
                                        memristor_info_dict=self.memristor_info_dict)
        # Corssbar for positive input and negative weight
        self.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for negative input and negative weight
        self.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)

        self.DAC_module_pos = DAC_Module(sim_params=sim_params, shape=self.shape,
                                    CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
        self.DAC_module_neg = DAC_Module(sim_params=sim_params, shape=self.shape,
                                    CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)

        if self.ADC_setting == 4:
            self.ADC_module_pos_pos = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
            self.ADC_module_neg_pos = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
            self.ADC_module_pos_neg = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
            self.ADC_module_neg_neg = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
        elif self.ADC_setting == 2:
            self.ADC_module_pos = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
            self.ADC_module_neg = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")


    def set_batch_size_mlp(self, batch_size) -> None:
        self.set_batch_size(batch_size)
        self.mem_pos_pos.set_batch_size(batch_size=batch_size)
        self.mem_neg_pos.set_batch_size(batch_size=batch_size)
        self.mem_pos_neg.set_batch_size(batch_size=batch_size)
        self.mem_neg_neg.set_batch_size(batch_size=batch_size)

        self.DAC_module_pos.set_batch_size(batch_size=batch_size)
        self.DAC_module_neg.set_batch_size(batch_size=batch_size)

        if self.ADC_setting == 4:
            self.ADC_module_pos_pos.set_batch_size(batch_size=batch_size)
            self.ADC_module_neg_pos.set_batch_size(batch_size=batch_size)
            self.ADC_module_pos_neg.set_batch_size(batch_size=batch_size)
            self.ADC_module_neg_neg.set_batch_size(batch_size=batch_size)
        elif self.ADC_setting == 2:
            self.ADC_module_pos.set_batch_size(batch_size=batch_size)
            self.ADC_module_neg.set_batch_size(batch_size=batch_size)
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")

        self.write_pulse_no = torch.zeros(batch_size, *self.shape, device=self.write_pulse_no.device)

        self.norm_ratio = torch.zeros(batch_size, device=self.norm_ratio.device) 



    def mapping_write_mlp(self, target_x):
        # Memristor reset first
        v_reset = self.memristor_luts[self.device_name]['V_reset']
        self.mem_v.fill_(v_reset)
        # Adopt large negative pulses to reset the memristor array
        self.DAC_module_pos.DAC_reset(mem_v=self.mem_v)
        self.DAC_module_neg.DAC_reset(mem_v=self.mem_v)

        self.mem_pos_pos.memristor_reset(mem_v=self.mem_v)
        self.mem_neg_pos.memristor_reset(mem_v=self.mem_v)
        self.mem_pos_neg.memristor_reset(mem_v=self.mem_v)
        self.mem_neg_neg.memristor_reset(mem_v=self.mem_v)

        # Transform target_x to [0, 1]
        self.norm_ratio = torch.max(torch.abs(target_x.reshape(target_x.shape[0], -1)), dim=1)[0]
        total_wr_cycle = self.memristor_luts[self.device_name]['total_no']
        write_voltage = self.memristor_luts[self.device_name]['voltage']
        counter = torch.ones_like(self.mem_v)

        # Positive weight write
        matrix_pos = torch.relu(target_x)
        # Vector to Pulse Serial
        self.write_pulse_no = self.m2v(matrix_pos / self.norm_ratio)
        # Matrix to memristor
        DAC_write_v = (self.write_pulse_no < (counter * total_wr_cycle)) * write_voltage
        self.DAC_module_pos.DAC_write(mem_v=DAC_write_v, mem_v_amp=write_voltage)
        # Memristor programming using multiple identical pulses (up to memristor states number)
        for t in range(total_wr_cycle):
            self.mem_v = ((counter * t) < self.write_pulse_no) * write_voltage
            self.mem_pos_pos.memristor_write(mem_v=self.mem_v, write_time=1, mem_v_amp=[write_voltage])
            self.mem_neg_pos.memristor_write(mem_v=self.mem_v, write_time=1, mem_v_amp=[write_voltage])

        # Negative weight write
        matrix_neg = torch.relu(target_x * -1)
        # Vector to Pulse Serial
        self.write_pulse_no = self.m2v(matrix_neg / self.norm_ratio)
        # Matrix to memristor
        DAC_write_v = (self.write_pulse_no < (counter * total_wr_cycle)) * write_voltage
        self.DAC_module_neg.DAC_write(mem_v=DAC_write_v, mem_v_amp=write_voltage)
        # Memristor programming using multiple identical pulses (up to memristor states number)
        for t in range(total_wr_cycle):
            self.mem_v = ((counter * t) < self.write_pulse_no) * write_voltage
            self.mem_pos_neg.memristor_write(mem_v=self.mem_v, write_time=1, mem_v_amp=[write_voltage])
            self.mem_neg_neg.memristor_write(mem_v=self.mem_v, write_time=1, mem_v_amp=[write_voltage])


    def mapping_read_mlp(self, target_v):
        # Get normalization ratio
        read_norm = torch.max(torch.abs(target_v), dim=1)[0]

        mem_v = (target_v / read_norm.unsqueeze(1)).unsqueeze(0)
        v_read_pos = self.DAC_module_pos.DAC_read(mem_v=mem_v, sgn='pos')
        v_read_neg = self.DAC_module_neg.DAC_read(mem_v=mem_v, sgn='neg')

        # memristor sequential read
        mem_i_sequence_pos_pos, _ = self.mem_pos_pos.memristor_read(mem_v=v_read_pos, read_time=1)
        mem_i_sequence_neg_pos, _ = self.mem_neg_pos.memristor_read(mem_v=v_read_neg, read_time=1)
        mem_i_sequence_pos_neg, _ = self.mem_pos_neg.memristor_read(mem_v=v_read_pos, read_time=1)
        mem_i_sequence_neg_neg, _ = self.mem_neg_neg.memristor_read(mem_v=v_read_neg, read_time=1)

        if self.ADC_setting == 4:
            ADC_mem_c_pos_pos = 1 / (1 / self.Goff + self.mem_pos_pos.total_wire_resistance)
            ADC_mem_c_neg_pos = 1 / (1 / self.Goff + self.mem_neg_pos.total_wire_resistance)
            ADC_mem_c_pos_neg = 1 / (1 / self.Goff + self.mem_pos_neg.total_wire_resistance)
            ADC_mem_c_neg_neg = 1 / (1 / self.Goff + self.mem_neg_neg.total_wire_resistance)
            mem_i_pos_pos = self.ADC_module_pos_pos.ADC_read(mem_i_sequence=mem_i_sequence_pos_pos, mem_c=ADC_mem_c_pos_pos, high_cut_ratio=1/self.ADC_setting)
            mem_i_neg_pos = self.ADC_module_neg_pos.ADC_read(mem_i_sequence=mem_i_sequence_neg_pos, mem_c=ADC_mem_c_neg_pos, high_cut_ratio=1/self.ADC_setting)
            mem_i_pos_neg = self.ADC_module_pos_neg.ADC_read(mem_i_sequence=mem_i_sequence_pos_neg, mem_c=ADC_mem_c_pos_neg, high_cut_ratio=1/self.ADC_setting)
            mem_i_neg_neg = self.ADC_module_neg_neg.ADC_read(mem_i_sequence=mem_i_sequence_neg_neg, mem_c=ADC_mem_c_neg_neg, high_cut_ratio=1/self.ADC_setting)
            mem_i = mem_i_pos_pos - mem_i_neg_pos - mem_i_pos_neg + mem_i_neg_neg
        elif self.ADC_setting == 2:
            ADC_mem_c_pos = 1 / (1 / self.Goff + self.mem_pos_pos.total_wire_resistance)
            ADC_mem_c_neg = 1 / (1 / self.Goff + self.mem_pos_neg.total_wire_resistance)
            mem_i_sequence_pos = mem_i_sequence_pos_pos + mem_i_sequence_neg_neg
            mem_i_pos = self.ADC_module_pos.ADC_read(mem_i_sequence_pos, mem_c=ADC_mem_c_pos, high_cut_ratio=1/self.ADC_setting)
            mem_i_sequence_neg = mem_i_sequence_neg_pos + mem_i_sequence_pos_neg
            mem_i_neg = self.ADC_module_pos.ADC_read(mem_i_sequence_neg, mem_c=ADC_mem_c_neg, high_cut_ratio=1/self.ADC_setting)
            mem_i = mem_i_pos - mem_i_neg
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")
        # Current to results
        self.mem_x_read = read_norm.unsqueeze(1) / (
                    2 ** self.input_bit - 1) * self.norm_ratio * self.trans_ratio * mem_i / self.v_read

        return self.mem_x_read.squeeze(0)


    def m2v(self, target_matrix):
        # Target_matrix ranging [0, 1]
        within_range = (target_matrix >= 0) & (target_matrix <= 1)
        assert torch.all(within_range), "The target Matrix Must be in the Range [0, 1]!"

        # Target x to target conductance
        target_c = target_matrix / self.trans_ratio + self.Gon

        # Get access to the look-up-table of the target memristor
        luts = self.memristor_luts[self.device_name]['conductance']

        # Find the nearest conductance value
        c_diff = torch.abs(torch.tensor(luts, device=target_c.device) - target_c.unsqueeze(3))
        nearest_pulse_no = torch.argmin(c_diff, dim=3)

        return nearest_pulse_no

    def update_SAF_mask(self) -> None:
        self.mem_pos_pos.update_SAF_mask()
        self.mem_pos_neg.update_SAF_mask()
        self.mem_neg_pos.update_SAF_mask()
        self.mem_neg_neg.update_SAF_mask()

    def total_energy_calculation(self) -> None:
        # language=rst
        """
        Calculate total energy for memristor-based architecture. Called when power is reported.
        """
        self.mem_pos_pos.total_energy_calculation()
        self.mem_neg_pos.total_energy_calculation()
        self.mem_pos_neg.total_energy_calculation()
        self.mem_neg_neg.total_energy_calculation()

        self.DAC_module_pos.DAC_energy_calculation(mem_t=self.mem_pos_pos.mem_t)
        self.DAC_module_neg.DAC_energy_calculation(mem_t=self.mem_neg_neg.mem_t)
        if self.ADC_setting == 4:
            self.ADC_module_pos_pos.ADC_energy_calculation(mem_t=self.mem_pos_pos.mem_t)
            self.ADC_module_neg_pos.ADC_energy_calculation(mem_t=self.mem_neg_pos.mem_t)
            self.ADC_module_pos_neg.ADC_energy_calculation(mem_t=self.mem_pos_neg.mem_t)
            self.ADC_module_neg_neg.ADC_energy_calculation(mem_t=self.mem_neg_neg.mem_t)
        elif self.ADC_setting == 2:
            self.ADC_module_pos.ADC_energy_calculation(mem_t=self.mem_pos_pos.mem_t)
            self.ADC_module_neg.ADC_energy_calculation(mem_t=self.mem_pos_neg.mem_t)
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")


        self.sim_power = {key: self.mem_pos_pos.power.sim_power[key] + self.mem_neg_pos.power.sim_power[key] +
                               self.mem_pos_neg.power.sim_power[key] + self.mem_neg_neg.power.sim_power[key] if key != 'time'
                          else self.mem_pos_pos.power.sim_power[key]
                          for key in self.mem_pos_pos.power.sim_power.keys()}

        self.sim_DAC_power = {key: self.DAC_module_pos.DAC_module_power.sim_power[key] + self.DAC_module_neg.DAC_module_power.sim_power[key]
                          for key in self.DAC_module_pos.DAC_module_power.sim_power.keys()}
        if self.ADC_setting == 4:
            self.sim_ADC_power = {key: self.ADC_module_pos_pos.ADC_module_power.sim_power[key] + self.ADC_module_neg_pos.ADC_module_power.sim_power[key] +
                                   self.ADC_module_pos_neg.ADC_module_power.sim_power[key] + self.ADC_module_neg_neg.ADC_module_power.sim_power[key]
                              for key in self.ADC_module_pos_pos.ADC_module_power.sim_power.keys()}
        elif self.ADC_setting == 2:
            self.sim_ADC_power = {key: self.ADC_module_pos.ADC_module_power.sim_power[key] + self.ADC_module_neg.ADC_module_power.sim_power[key]
                              for key in self.ADC_module_pos.ADC_module_power.sim_power.keys()}
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")

        self.sim_periph_power = {**self.sim_DAC_power, **self.sim_ADC_power}


    def total_area_calculation(self) -> None:
        # language=rst
        """
        Calculate total area for memristor-based architecture. Called when power is reported.
        """
        self.sim_mem_area = self.mem_pos_pos.area.array_area + self.mem_neg_pos.area.array_area + self.mem_pos_neg.area.array_area + self.mem_neg_neg.area.array_area

        DAC_height_row_pos, DAC_width_row_pos, DAC_height_col_pos, DAC_width_col_pos, sim_switch_matrix_row_area_pos, sim_switch_matrix_col_area_pos = self.DAC_module_pos.DAC_module_area.DAC_module_cal_area()
        DAC_height_row_neg, DAC_width_row_neg, DAC_height_col_neg, DAC_width_col_neg, sim_switch_matrix_row_area_neg, sim_switch_matrix_col_area_neg = self.DAC_module_neg.DAC_module_area.DAC_module_cal_area()
        sim_switch_matrix_row_area = sim_switch_matrix_row_area_neg + sim_switch_matrix_row_area_pos
        sim_switch_matrix_col_area = sim_switch_matrix_col_area_neg + sim_switch_matrix_col_area_pos

        if self.ADC_setting == 4:
            ADC_height_pos_pos, ADC_width_pos_pos, sim_shiftadd_area_pos_pos, sim_SarADC_area_pos_pos = self.ADC_module_pos_pos.ADC_module_area.ADC_module_cal_area()
            ADC_height_neg_pos, ADC_width_neg_pos, sim_shiftadd_area_neg_pos, sim_SarADC_area_neg_pos = self.ADC_module_neg_pos.ADC_module_area.ADC_module_cal_area()
            ADC_height_pos_neg, ADC_width_pos_neg, sim_shiftadd_area_pos_neg, sim_SarADC_area_pos_neg = self.ADC_module_pos_neg.ADC_module_area.ADC_module_cal_area()
            ADC_height_neg_neg, ADC_width_neg_neg, sim_shiftadd_area_neg_neg, sim_SarADC_area_neg_neg = self.ADC_module_neg_neg.ADC_module_area.ADC_module_cal_area()
            sim_SarADC_area = sim_SarADC_area_neg_neg + sim_SarADC_area_neg_pos + sim_SarADC_area_pos_neg + sim_SarADC_area_pos_pos
            sim_shiftadd_area = sim_shiftadd_area_neg_neg + sim_shiftadd_area_neg_pos + sim_shiftadd_area_pos_neg + sim_shiftadd_area_pos_pos
        elif self.ADC_setting == 2:
            ADC_height_pos, ADC_width_pos, sim_shiftadd_area_pos, sim_SarADC_area_pos = self.ADC_module_pos.ADC_module_area.ADC_module_cal_area()
            ADC_height_neg, ADC_width_neg, sim_shiftadd_area_neg, sim_SarADC_area_neg = self.ADC_module_neg.ADC_module_area.ADC_module_cal_area()
            sim_shiftadd_area = sim_shiftadd_area_neg + sim_shiftadd_area_pos
            sim_SarADC_area = sim_SarADC_area_neg + sim_SarADC_area_pos
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")

        periph_total_area = sim_switch_matrix_row_area + sim_switch_matrix_col_area + sim_shiftadd_area + sim_SarADC_area
        self.sim_periph_area = {'sim_switch_matrix_row_area': sim_switch_matrix_row_area,
                             'sim_switch_matrix_col_area': sim_switch_matrix_col_area,
                             'sim_shiftadd_area': sim_shiftadd_area, 'sim_SarADC_area': sim_SarADC_area,
                             'sim_total_periph_area': periph_total_area}

        if self.ADC_setting == 4:
            total_height_pos_pos = max(self.mem_pos_pos.length_col + ADC_height_pos_pos + DAC_height_col_pos, DAC_height_row_pos)
            total_width_pos_pos = DAC_width_row_pos + max(self.mem_pos_pos.length_row, DAC_width_col_pos, ADC_width_pos_pos)
            total_height_neg_pos = self.mem_neg_pos.length_col + ADC_height_neg_pos
            total_width_neg_pos = max(self.mem_neg_pos.length_row, ADC_width_neg_pos)
            total_height_pos_neg = max(self.mem_pos_neg.length_col + ADC_height_pos_neg + DAC_height_col_neg, DAC_height_row_neg)
            total_width_pos_neg = DAC_width_row_neg + max(self.mem_pos_neg.length_row, DAC_width_col_neg, ADC_width_pos_neg)
            total_height_neg_neg = self.mem_neg_neg.length_col + ADC_height_neg_neg
            total_width_neg_neg = max(self.mem_neg_neg.length_row, ADC_width_neg_neg)
        elif self.ADC_setting == 2:
            total_height_pos_pos = max(self.mem_pos_pos.length_col + ADC_height_pos + DAC_height_col_pos, DAC_height_row_pos)
            total_width_pos_pos = DAC_width_row_pos + max(self.mem_pos_pos.length_row, DAC_width_col_pos, ADC_width_pos)
            total_height_neg_pos = self.mem_neg_pos.length_col
            total_width_neg_pos = self.mem_neg_pos.length_row
            total_height_pos_neg = max(self.mem_pos_neg.length_col + ADC_height_neg + DAC_height_col_neg, DAC_height_row_neg)
            total_width_pos_neg = DAC_width_row_neg + max(self.mem_pos_neg.length_row, DAC_width_col_neg, ADC_width_neg)
            total_height_neg_neg = self.mem_neg_neg.length_col
            total_width_neg_neg = self.mem_neg_neg.length_row
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")

        self.sim_total_area = total_height_pos_pos * total_width_pos_pos + total_height_pos_neg * total_width_pos_neg \
                    + total_height_neg_pos * total_width_neg_pos + total_height_neg_neg * total_width_neg_neg

        self.sim_area = {'sim_mem_area':self.sim_mem_area,
                         'sim_periph_area':periph_total_area,
                         'sim_total_area':self.sim_total_area,
                         'sim_used_area_ratio':(self.sim_mem_area+periph_total_area)/self.sim_total_area}


class CNNMapping(Mapping):
    # language=rst
    """
    Mapping convolutional layers (Conv2D) to memristor arrays.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the memristor array.
        """
        super().__init__(
            sim_params=sim_params,
            shape=shape
        )

        self.register_buffer("norm_ratio", torch.Tensor())
        # self.register_buffer("write_pulse_no", torch.Tensor())

        # Corssbar for positive input and positive weight
        self.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for negative input and positive weight
        self.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for positive input and negative weight
        self.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for negative input and negative weight
        self.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)

        self.DAC_module_pos = DAC_Module(sim_params=sim_params, shape=self.shape,
                                    CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
        self.DAC_module_neg = DAC_Module(sim_params=sim_params, shape=self.shape,
                                    CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)

        if self.ADC_setting == 4:
            self.ADC_module_pos_pos = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
            self.ADC_module_neg_pos = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
            self.ADC_module_pos_neg = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
            self.ADC_module_neg_neg = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
        else:
            raise Exception("Only 4-set ADC are supported!")


    def set_batch_size_cnn(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when memristor is used to mapping convolutional layers (Conv2D).
    
        :param batch_size: Mini-batch size.
        """
        self.set_batch_size(batch_size)
        self.mem_pos_pos.set_batch_size(batch_size=batch_size)
        self.mem_neg_pos.set_batch_size(batch_size=batch_size)
        self.mem_pos_neg.set_batch_size(batch_size=batch_size)
        self.mem_neg_neg.set_batch_size(batch_size=batch_size)

        self.DAC_module_pos.set_batch_size(batch_size=batch_size)
        self.DAC_module_neg.set_batch_size(batch_size=batch_size)

        if self.ADC_setting == 4:
            self.ADC_module_pos_pos.set_batch_size(batch_size=batch_size)
            self.ADC_module_neg_pos.set_batch_size(batch_size=batch_size)
            self.ADC_module_pos_neg.set_batch_size(batch_size=batch_size)
            self.ADC_module_neg_neg.set_batch_size(batch_size=batch_size)
        else:
            raise Exception("Only 4-set ADC are supported!")

        # self.write_pulse_no = torch.zeros(batch_size, *self.shape, device=self.mem_v.device)
        self.norm_ratio_pos = torch.zeros(batch_size, device=self.norm_ratio.device)
        self.norm_ratio_neg = torch.zeros(batch_size, device=self.norm_ratio.device)


    def mapping_write_cnn(self, target_x):
        # Memristor reset first
        v_reset = self.memristor_luts[self.device_name]['V_reset']
        self.mem_v.fill_(v_reset)
        # Adopt large negative pulses to reset the memristor array
        self.DAC_module_pos.DAC_reset(mem_v=self.mem_v)
        self.DAC_module_neg.DAC_reset(mem_v=self.mem_v)

        self.mem_pos_pos.memristor_reset(mem_v=self.mem_v)
        self.mem_neg_pos.memristor_reset(mem_v=self.mem_v)
        self.mem_pos_neg.memristor_reset(mem_v=self.mem_v)
        self.mem_neg_neg.memristor_reset(mem_v=self.mem_v)

        # Transform target_x to [0, 1]
        self.norm_ratio_pos = torch.max(torch.relu(target_x).reshape(target_x.shape[0], -1), dim=1)[0]
        self.norm_ratio_neg = torch.max(torch.relu(target_x * -1).reshape(target_x.shape[0], -1), dim=1)[0]
        total_wr_cycle = self.memristor_luts[self.device_name]['total_no']
        write_voltage = self.memristor_luts[self.device_name]['voltage']
        counter = torch.ones_like(self.mem_v)

        # Positive weight write
        matrix_pos = torch.relu(target_x)
        # Vector to Pulse Serial
        write_pulse_no = self.m2v(matrix_pos / self.norm_ratio_pos)
        # Matrix to memristor
        DAC_write_v = (write_pulse_no < (counter * total_wr_cycle)) * write_voltage
        self.DAC_module_pos.DAC_write(mem_v=DAC_write_v, mem_v_amp=write_voltage)
        # Memristor programming using multiple identical pulses (up to memristor states number)
        for t in range(total_wr_cycle):
            self.mem_v = ((counter * t) < write_pulse_no) * write_voltage
            self.mem_pos_pos.memristor_write(mem_v=self.mem_v, write_time=1, mem_v_amp=[write_voltage])
            self.mem_neg_pos.memristor_write(mem_v=self.mem_v, write_time=1, mem_v_amp=[write_voltage])

        # Negative weight write
        matrix_neg = torch.relu(target_x * -1)
        # Vector to Pulse Serial
        write_pulse_no = self.m2v(matrix_neg / self.norm_ratio_neg)
        # Matrix to memristor
        DAC_write_v = (write_pulse_no < (counter * total_wr_cycle)) * write_voltage
        self.DAC_module_neg.DAC_write(mem_v=DAC_write_v, mem_v_amp=write_voltage)
        # Memristor programming using multiple identical pulses (up to memristor states number)
        for t in range(total_wr_cycle):
            self.mem_v = ((counter * t) < write_pulse_no) * write_voltage
            self.mem_pos_neg.memristor_write(mem_v=self.mem_v, write_time=1, mem_v_amp=[write_voltage])
            self.mem_neg_neg.memristor_write(mem_v=self.mem_v, write_time=1, mem_v_amp=[write_voltage])


    def mapping_read_cnn(self, target_v):
        # Get normalization ratio
        read_norm = torch.max(torch.abs(target_v), dim=1)[0]
        target_v = target_v.unsqueeze(0)
        mem_v = target_v / read_norm.unsqueeze(0).unsqueeze(2)
        v_read_pos = self.DAC_module_pos.DAC_read(mem_v=mem_v, sgn='pos')
        v_read_neg = self.DAC_module_neg.DAC_read(mem_v=mem_v, sgn='neg')

        # memristor sequential read
        mem_i_sequence_pos_pos, _ = self.mem_pos_pos.memristor_read(mem_v=v_read_pos, read_time=1)
        mem_i_sequence_neg_pos, _ = self.mem_neg_pos.memristor_read(mem_v=v_read_neg, read_time=1)
        mem_i_sequence_pos_neg, _ = self.mem_pos_neg.memristor_read(mem_v=v_read_pos, read_time=1)
        mem_i_sequence_neg_neg, _ = self.mem_neg_neg.memristor_read(mem_v=v_read_neg, read_time=1)

        if self.ADC_setting == 4:
            ADC_mem_c_pos_pos = 1 / (1 / self.Goff + self.mem_pos_pos.total_wire_resistance)
            ADC_mem_c_neg_pos = 1 / (1 / self.Goff + self.mem_neg_pos.total_wire_resistance)
            ADC_mem_c_pos_neg = 1 / (1 / self.Goff + self.mem_pos_neg.total_wire_resistance)
            ADC_mem_c_neg_neg = 1 / (1 / self.Goff + self.mem_neg_neg.total_wire_resistance)
            mem_i_pos_pos = self.ADC_module_pos_pos.ADC_read(mem_i_sequence=mem_i_sequence_pos_pos, mem_c=ADC_mem_c_pos_pos, high_cut_ratio=2/self.ADC_setting)
            mem_i_neg_pos = self.ADC_module_neg_pos.ADC_read(mem_i_sequence=mem_i_sequence_neg_pos, mem_c=ADC_mem_c_neg_pos, high_cut_ratio=2/self.ADC_setting)
            mem_i_pos_neg = self.ADC_module_pos_neg.ADC_read(mem_i_sequence=mem_i_sequence_pos_neg, mem_c=ADC_mem_c_pos_neg, high_cut_ratio=2/self.ADC_setting)
            mem_i_neg_neg = self.ADC_module_neg_neg.ADC_read(mem_i_sequence=mem_i_sequence_neg_neg, mem_c=ADC_mem_c_neg_neg, high_cut_ratio=2/self.ADC_setting)
            mem_i_pos = mem_i_pos_pos - mem_i_neg_pos
            mem_i_neg = mem_i_pos_neg - mem_i_neg_neg
        else:
            raise Exception("Only 4-set ADC are supported!")

        mem_i_pos = mem_i_pos_pos - mem_i_neg_pos
        mem_i_neg = mem_i_pos_neg - mem_i_neg_neg

        # Current to results
        self.mem_x_read = self.norm_ratio_pos * self.trans_ratio * (
                    read_norm.unsqueeze(1) / (2 ** self.input_bit - 1) / self.v_read * mem_i_pos - (
                        target_v.squeeze().sum(dim=1) * self.Gon).unsqueeze(0).unsqueeze(2))

        # Current to results
        self.mem_x_read -= self.norm_ratio_neg * self.trans_ratio * (
                    read_norm.unsqueeze(1) / (2 ** self.input_bit - 1) / self.v_read * mem_i_neg - (
                        target_v.squeeze().sum(dim=1) * self.Gon).unsqueeze(0).unsqueeze(2))

        return self.mem_x_read.squeeze(0)


    def m2v(self, target_matrix):
        # Target_matrix ranging [0, 1]
        within_range = (target_matrix >= 0) & (target_matrix <= 1)
        assert torch.all(within_range), "The target Matrix Must be in the Range [0, 1]!"

        # Target x to target conductance
        target_c = target_matrix / self.trans_ratio + self.Gon

        # Get access to the look-up-table of the target memristor
        luts = self.memristor_luts[self.device_name]['conductance']

        # Find the nearest conductance value
        len_luts = len(luts)
        section = 1 + (len_luts - 1) // 100
        seg_len = len_luts // section
        nearest_pulse_no = torch.zeros_like(target_c)
        for i in range(section):
            c_diff = torch.abs(torch.tensor(luts[(i * seg_len):(i * seg_len + seg_len + 1)], device=target_c.device) - target_c.unsqueeze(3))
            nearest_pulse_no += torch.argmin(c_diff, dim=3)
            c_diff = None

        return nearest_pulse_no


    def update_SAF_mask(self) -> None:
        self.mem_pos_pos.update_SAF_mask()
        self.mem_pos_neg.update_SAF_mask()
        self.mem_neg_pos.update_SAF_mask()
        self.mem_neg_neg.update_SAF_mask()


    def total_energy_calculation(self) -> None:
        # language=rst
        """
        Calculate total energy for memristor-based architecture. Called when power is reported.
        """
        self.mem_pos_pos.total_energy_calculation()
        self.mem_neg_pos.total_energy_calculation()
        self.mem_pos_neg.total_energy_calculation()
        self.mem_neg_neg.total_energy_calculation()

        self.DAC_module_pos.DAC_energy_calculation(mem_t=self.mem_pos_pos.mem_t)
        self.DAC_module_neg.DAC_energy_calculation(mem_t=self.mem_neg_neg.mem_t)
        if self.ADC_setting == 4:
            self.ADC_module_pos_pos.ADC_energy_calculation(mem_t=self.mem_pos_pos.mem_t)
            self.ADC_module_neg_pos.ADC_energy_calculation(mem_t=self.mem_neg_pos.mem_t)
            self.ADC_module_pos_neg.ADC_energy_calculation(mem_t=self.mem_pos_neg.mem_t)
            self.ADC_module_neg_neg.ADC_energy_calculation(mem_t=self.mem_neg_neg.mem_t)
        else:
            raise Exception("Only 4-set ADC are supported!")

        self.sim_power = {key: self.mem_pos_pos.power.sim_power[key] + self.mem_neg_pos.power.sim_power[key] +
                               self.mem_pos_neg.power.sim_power[key] + self.mem_neg_neg.power.sim_power[key] if key != 'time'
                          else self.mem_pos_pos.power.sim_power[key]
                          for key in self.mem_pos_pos.power.sim_power.keys()}

        self.sim_DAC_power = {key: self.DAC_module_pos.DAC_module_power.sim_power[key] + self.DAC_module_neg.DAC_module_power.sim_power[key]
                          for key in self.DAC_module_pos.DAC_module_power.sim_power.keys()}
        if self.ADC_setting == 4:
            self.sim_ADC_power = {key: self.ADC_module_pos_pos.ADC_module_power.sim_power[key] + self.ADC_module_neg_pos.ADC_module_power.sim_power[key] +
                                   self.ADC_module_pos_neg.ADC_module_power.sim_power[key] + self.ADC_module_neg_neg.ADC_module_power.sim_power[key]
                              for key in self.ADC_module_pos_pos.ADC_module_power.sim_power.keys()}
        else:
            raise Exception("Only 4-set ADC are supported!")

        self.sim_periph_power = {**self.sim_DAC_power, **self.sim_ADC_power}


    def total_area_calculation(self) -> None:
        # language=rst
        """
        Calculate total area for memristor-based architecture. Called when power is reported.
        """
        self.sim_mem_area = self.mem_pos_pos.area.array_area + self.mem_neg_pos.area.array_area + self.mem_pos_neg.area.array_area + self.mem_neg_neg.area.array_area

        DAC_height_row_pos, DAC_width_row_pos, DAC_height_col_pos, DAC_width_col_pos, sim_switch_matrix_row_area_pos, sim_switch_matrix_col_area_pos = self.DAC_module_pos.DAC_module_area.DAC_module_cal_area()
        DAC_height_row_neg, DAC_width_row_neg, DAC_height_col_neg, DAC_width_col_neg, sim_switch_matrix_row_area_neg, sim_switch_matrix_col_area_neg = self.DAC_module_neg.DAC_module_area.DAC_module_cal_area()
        sim_switch_matrix_row_area = sim_switch_matrix_row_area_neg + sim_switch_matrix_row_area_pos
        sim_switch_matrix_col_area = sim_switch_matrix_col_area_neg + sim_switch_matrix_col_area_pos

        if self.ADC_setting == 4:
            ADC_height_pos_pos, ADC_width_pos_pos, sim_shiftadd_area_pos_pos, sim_SarADC_area_pos_pos = self.ADC_module_pos_pos.ADC_module_area.ADC_module_cal_area()
            ADC_height_neg_pos, ADC_width_neg_pos, sim_shiftadd_area_neg_pos, sim_SarADC_area_neg_pos = self.ADC_module_neg_pos.ADC_module_area.ADC_module_cal_area()
            ADC_height_pos_neg, ADC_width_pos_neg, sim_shiftadd_area_pos_neg, sim_SarADC_area_pos_neg = self.ADC_module_pos_neg.ADC_module_area.ADC_module_cal_area()
            ADC_height_neg_neg, ADC_width_neg_neg, sim_shiftadd_area_neg_neg, sim_SarADC_area_neg_neg = self.ADC_module_neg_neg.ADC_module_area.ADC_module_cal_area()
            sim_SarADC_area = sim_SarADC_area_neg_neg + sim_SarADC_area_neg_pos + sim_SarADC_area_pos_neg + sim_SarADC_area_pos_pos
            sim_shiftadd_area = sim_shiftadd_area_neg_neg + sim_shiftadd_area_neg_pos + sim_shiftadd_area_pos_neg + sim_shiftadd_area_pos_pos
        else:
            raise Exception("Only 4-set ADC are supported!")

        periph_total_area = sim_switch_matrix_row_area + sim_switch_matrix_col_area + sim_shiftadd_area + sim_SarADC_area
        self.sim_periph_area = {'sim_switch_matrix_row_area': sim_switch_matrix_row_area,
                             'sim_switch_matrix_col_area': sim_switch_matrix_col_area,
                             'sim_shiftadd_area': sim_shiftadd_area, 'sim_SarADC_area': sim_SarADC_area,
                             'sim_total_periph_area': periph_total_area}

        if self.ADC_setting == 4:
            total_height_pos_pos = max(self.mem_pos_pos.length_col + ADC_height_pos_pos + DAC_height_col_pos, DAC_height_row_pos)
            total_width_pos_pos = DAC_width_row_pos + max(self.mem_pos_pos.length_row, DAC_width_col_pos, ADC_width_pos_pos)
            total_height_neg_pos = self.mem_neg_pos.length_col + ADC_height_neg_pos
            total_width_neg_pos = max(self.mem_neg_pos.length_row, ADC_width_neg_pos)
            total_height_pos_neg = max(self.mem_pos_neg.length_col + ADC_height_pos_neg + DAC_height_col_neg, DAC_height_row_neg)
            total_width_pos_neg = DAC_width_row_neg + max(self.mem_pos_neg.length_row, DAC_width_col_neg, ADC_width_pos_neg)
            total_height_neg_neg = self.mem_neg_neg.length_col + ADC_height_neg_neg
            total_width_neg_neg = max(self.mem_neg_neg.length_row, ADC_width_neg_neg)
        else:
            raise Exception("Only 4-set ADC are supported!")

        self.sim_total_area = total_height_pos_pos * total_width_pos_pos + total_height_pos_neg * total_width_pos_neg \
                    + total_height_neg_pos * total_width_neg_pos + total_height_neg_neg * total_width_neg_neg

        self.sim_area = {'sim_mem_area':self.sim_mem_area,
                         'sim_periph_area':periph_total_area,
                         'sim_total_area':self.sim_total_area,
                         'sim_used_area_ratio':(self.sim_mem_area+periph_total_area)/self.sim_total_area}


