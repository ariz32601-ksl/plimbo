#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright 2018-2019 by Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
Abstract base classes for high-level simulator objects.
'''

from abc import ABCMeta, abstractmethod
import numpy as np
from collections import OrderedDict
import copy
# --- SOVEREIGN AUTOMATED CORRECTION ---
try:
    from betse.util.serial.pickle import pickles  # Modern BETSE path
except ImportError:
    from betse.lib.pickle import pickles         # Legacy fallback
from betse.science.parameters import Parameters
from betse.util.type.types import NumpyArrayType


class PlanariaGRNABC(object, metaclass=ABCMeta):
    """
    Abstract base class of all objects modelling a GRN.

    A BETSE config file is used to define paths for saving image and data
    exports.
    """

    def __init__(self, *args, **kwargs):

        self.model_init(*args, **kwargs)

    def model_init(self, config_filename, pdict = None, xscale=1.0, new_mesh=False,
                 verbose=False):

        if pdict is None: # default parameters

            self.pdict = OrderedDict({ #params optimized for 2D

                # Beta cat parameters
                'r_bc': 2.5e-3,  #
                'd_bc': 1.0e-6,  #
                'd_bc_deg': 5.0e-3,  #
                'K_bc_apc': 0.5,
                'n_bc_apc': 2.0,
                'K_bc_camp': 1.0,
                'n_bc_camp': 2.0,
                'D_bc': 1.0e-12,

                # ERK parameters
                'K_erk_bc': 30.0,  #
                'n_erk_bc': 2.0,  #

                # APC parameters
                'K_apc_wnt': 30.0,  #
                'n_apc_wnt': 2.0,  #

                # Hedgehog parameters:
                'r_hh': 5.0e-3,  #
                'd_hh': 5.0e-6,  #
                'D_hh': 1.5e-11,  #
                'u_hh': 3.0e-8,  #

                # Wnt1 parameters
                'r_wnt': 1.0e-2,  #
                'd_wnt': 1.0e-5,  #
                'K_wnt_notum': 0.5,  #
                'n_wnt_notum': 2.5,  # 2.5
                'D_wnt': 5.0e-12,  #
                'd_wnt_deg_notum': 1.0e-4,  # 1.0e-4
                'd_wnt_deg_ptc': 7.5e-5,  #
                'K_wnt_hh': 300.0,  #
                'n_wnt_hh': 2.5,  # 2.5

                # NRF parameters
                'r_nrf': 7.5e-3,  #
                'd_nrf': 7.5e-6,  #
                'K_nrf_bc': 150.0,  # 150
                'n_nrf_bc': 2.0,  # 2.0
                'D_nrf': 1.5e-11,  #
                'u_nrf': -6.0e-8,  #

                # Notum parameters
                'r_notum': 2.5e-3,  #
                'd_notum': 2.5e-3,  #
                'K_notum_nrf': 250.0,  # 250, 150
                'n_notum_nrf': 2.0,  # 2.0, 1.0
                'D_notum': 1.5e-11,  #

                # Markov model parameters:
                'C1': 0.75,  # ERK constant to modulate head formation
                'K1': 0.05,

                'C2': 300.0,  # Beta-catenin concentration to modulate tail formation
                'K2': 40.0,

                'Beta_B': 5.0e-3,  # head/tail tissue decay time constant

                'hdac_to': 90.0 * 3600,  # 96 time at which hdac stops growing
                'hdac_ts': 12.0 * 3600,  # time period over which hdac stops growing

                'nd_min': 0.4, # minimum value of the nerve map
                'nd_max': 1.0  # maximu value of the nerve map

            })



        else:
            self.pdict = pdict

        # BETSE parameters object:
        self.p = Parameters.make(config_filename)

        self.verbose = verbose

        self.x_scale = xscale

        self.make_mesh()

        self.prime_model()

        # Initialize the transport field and nerve density:
        self.load_transport_field()

        self.runtype = 'init'

        self.model_has_been_cut = False

        if self.verbose is True:
            print("Successfully generated model!")


    def prime_model(self):

        if self.verbose is True:
            print("Initializing parameters and variables...")

        # tags for easy reference to concentrations of the model:
        self.conc_tags = ['β-Cat', 'Erk', 'Wnt', 'Hh', 'NRF', 'Notum', 'APC', 'cAMP', 'Head', 'Tail']

        # Initialize the master molecules handlers to null values to avoid plot/animation issues:
        self.molecules_time = OrderedDict()
        self.molecules_time2 = OrderedDict()
        self.molecules_sim_time = OrderedDict()

        for tag in self.conc_tags:
            self.molecules_time[tag] = np.zeros(self.cdl)
            self.molecules_time2[tag] = np.zeros(self.cdl)
            self.molecules_sim_time[tag] = np.zeros(self.cdl)

        # Default RNAi keys:
        self.RNAi_defaults = {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1,
         'wnt': 1, 'hh': 1, 'camp': 1,'dynein': 1, 'kinesin':1, 'ptc':1}

        self.init_plots()

        # Beta cat parameters
        self.r_bc = self.pdict['r_bc']
        self.d_bc = self.pdict['d_bc']
        self.d_bc_deg = self.pdict['d_bc_deg']
        self.K_bc_apc = self.pdict['K_bc_apc']
        self.n_bc_apc = self.pdict['n_bc_apc']
        self.K_bc_camp = self.pdict['K_bc_camp']
        self.n_bc_camp = self.pdict['n_bc_camp']
        self.D_bc = self.pdict['D_bc']
        # self.u_bc = self.pdict['u_bc']

        self.c_BC = np.ones(self.cdl)
        self.c_BC_time = []

        # ERK parameters
        self.K_erk_bc = self.pdict['K_erk_bc']
        self.K_erk_bc = self.pdict['K_erk_bc']
        self.n_erk_bc = self.pdict['n_erk_bc']

        self.c_ERK = np.zeros(self.cdl)
        self.c_ERK_time = []

        # APC parameters
        self.K_apc_wnt = self.pdict['K_apc_wnt']
        self.n_apc_wnt = self.pdict['n_apc_wnt']

        # Hedgehog parameters:
        self.r_hh = self.pdict['r_hh']
        self.d_hh = self.pdict['d_hh']
        self.D_hh = self.pdict['D_hh']
        self.u_hh = self.pdict['u_hh']

        self.c_HH = np.zeros(self.cdl)
        self.c_HH_time = []

        # Wnt parameters
        self.r_wnt = self.pdict['r_wnt']
        self.d_wnt = self.pdict['d_wnt']
        self.d_wnt_deg_notum = self.pdict['d_wnt_deg_notum']
        self.d_wnt_deg_ptc = self.pdict['d_wnt_deg_ptc']
        self.D_wnt = self.pdict['D_wnt']
        self.K_wnt_notum = self.pdict['K_wnt_notum']
        self.n_wnt_notum = self.pdict['n_wnt_notum']
        self.K_wnt_hh = self.pdict['K_wnt_hh']
        self.n_wnt_hh = self.pdict['n_wnt_hh']
        # self.K_wnt_camp = self.pdict['K_wnt_camp']
        # self.n_wnt_camp = self.pdict['n_wnt_camp']

        self.c_WNT = np.zeros(self.cdl)
        self.c_WNT_time = []

        # Notum regulating factor (NRF) parameters
        self.r_nrf = self.pdict['r_nrf']
        self.d_nrf = self.pdict['d_nrf']
        self.K_nrf_bc = self.pdict['K_nrf_bc']
        self.n_nrf_bc = self.pdict['n_nrf_bc']
        self.D_nrf = self.pdict['D_nrf']
        self.u_nrf = self.pdict['u_nrf']

        self.c_NRF = np.zeros(self.cdl)
        self.c_NRF_time = []

        # Notum parameters
        self.D_notum = self.pdict['D_notum']
        self.r_notum = self.pdict['r_notum']
        self.d_notum = self.pdict['d_notum']
        self.K_notum_nrf = self.pdict['K_notum_nrf']
        self.n_notum_nrf = self.pdict['n_notum_nrf']

        self.c_Notum = np.zeros(self.cdl)
        self.c_Notum_time = []

        # cAMP parameters
        self.c_cAMP = np.ones(self.cdl)

        # Markov model parameters:
        self.C1 = self.pdict['C1'] # ERK constant to modulate head formation
        self.K1 = self.pdict['K1']
        self.C2 = self.pdict['C2']  # Beta-catenin concentration to modulate tail formation
        self.K2 = self.pdict['K2']
        self.beta_B = self.pdict['Beta_B']  # head/tail tissue decay time constant
        self.alpha_BH = 1/(1 + np.exp(-(self.c_ERK - self.C1)/self.K1)) # init transition constant blastema to head
        self.alpha_BT = 1/(1 + np.exp(-(self.c_BC - self.C2)/self.K2)) # init transition constant blastema to tail

        # initialize Markov model probabilities:
        self.Head = np.zeros(self.cdl) # head
        self.Tail = np.zeros(self.cdl) # tail
        self.Blast = np.ones(self.cdl) # blastema or stem cells

        # Remodeling time window parameters:
        self.hdac_to = self.pdict['hdac_to']
        self.hdac_ts = self.pdict['hdac_ts']

        # offset to nerve map used to help match 1D and 2D cases by setting same range on transport map:
        self.n_min = self.pdict['nd_min']
        self.n_max = self.pdict['nd_max']


    @abstractmethod
    def make_mesh(self):

        pass

    @abstractmethod
    def cut_cells(self):

        pass

    @abstractmethod
    def scale_cells(self, x_scale):
        pass

    @abstractmethod
    def run_markov(self, ti):
        """
        Updates the Markov model in time

        :return:

        """

        pass

    @abstractmethod
    def load_transport_field(self):

        pass

    # GRN Running functions---------------------------------------
    @abstractmethod
    def update_bc(self, rnai_bc=1.0, rnai_apc=1.0, rnai_camp = 1.0, rnai_erk=1.0) -> NumpyArrayType:
        """
        Method describing change in beta-cat levels in space and time.
        """

        pass

    @abstractmethod
    def update_nrf(self, dynein=1.0) -> NumpyArrayType:
        """
        Method describing change in NRF levels in space and time.
        """

        pass

    @abstractmethod
    def update_notum(self, rnai=1.0) -> NumpyArrayType:
        """
        Method describing change in Notum levels in space and time.
        """

        pass

    @abstractmethod
    def update_wnt(self, rnai=1.0, rnai2 = 1.0) -> NumpyArrayType:
        """
        Method describing change in Wnt1 and Wnt 11 levels in space and time.
        """

        pass

    @abstractmethod
    def update_hh(self, rnai=1.0, kinesin=1.0) -> NumpyArrayType:
        """
        Method describing change in Hedgehog levels in space and time.
        """

        pass

    #-----------------------------------

    def clear_cache_init(self):

        self.c_BC_time = []
        self.c_ERK_time = []
        self.c_WNT_time = []
        self.c_HH_time = []
        self.c_NRF_time = []
        self.c_Notum_time = []

        self.Head_time = []
        self.Tail_time = []
        self.Blast_time = []

        # self.hdac_time = []

    def clear_cache_reinit(self):

        self.c_BC_time2 = []
        self.c_ERK_time2 = []
        self.c_WNT_time2 = []
        self.c_HH_time2 = []
        self.c_NRF_time2 = []
        self.c_Notum_time2 = []

        # self.hdac_time2 = []

        self.Head_time2 = []
        self.Tail_time2 = []
        self.Blast_time2 = []

    def clear_cache_sim(self):

        self.c_BC_sim_time = []
        self.c_ERK_sim_time = []
        self.c_WNT_sim_time = []
        self.c_HH_sim_time = []
        self.c_NRF_sim_time = []
        self.c_Notum_sim_time = []

        # self.hdac_sim_time = []

        self.Head_sim_time = []
        self.Tail_sim_time = []
        self.Blast_sim_time = []

    def run_loop(self,
                 knockdown=None):

        if knockdown is None:
            knockdown = self.RNAi_defaults

        for tt in self.time:
            # time update beta-catenin and erk signalling:
            delta_bc = self.update_bc(rnai_bc=knockdown['bc'], rnai_camp = knockdown['camp'],
                                      rnai_apc =knockdown['apc'], rnai_erk=knockdown['erk']) * self.dt

            delta_wnt = self.update_wnt(rnai_wnt=knockdown['wnt'], rnai_ptc = knockdown['ptc'])*self.dt  # time update wnt

            delta_hh = self.update_hh(rnai=knockdown['hh'],
                                      kinesin=knockdown['kinesin']) * self.dt  # time update hh

            delta_nrf = self.update_nrf(dynein=knockdown['dynein']) * self.dt  # update NRF

            delta_notum = self.update_notum(rnai=knockdown['notum']) * self.dt  # time update Notum

            self.c_BC += delta_bc  # time update beta-catenin
            self.c_WNT += delta_wnt  # time update Wnt
            self.c_HH += delta_hh  # time update Hh
            self.c_NRF += delta_nrf  # time update NRF
            self.c_Notum += delta_notum  # time update Notum

            # update the Markov model:
            self.run_markov(tt)

            if tt in self.tsample:

                if self.runtype == 'init':

                    self.c_BC_time.append(self.c_BC * 1)
                    self.c_WNT_time.append(self.c_WNT * 1)
                    self.c_HH_time.append(self.c_HH * 1)
                    self.c_Notum_time.append(self.c_Notum * 1)
                    self.c_NRF_time.append(self.c_NRF * 1)
                    self.c_ERK_time.append(self.c_ERK * 1)

                    # self.hdac_time.append(self.hdac*1)

                    self.Head_time.append(self.Head*1)
                    self.Tail_time.append(self.Tail*1)
                    self.Blast_time.append(self.Blast*1)

                elif self.runtype == 'reinit':

                    self.c_BC_time2.append(self.c_BC * 1)
                    self.c_WNT_time2.append(self.c_WNT * 1)
                    self.c_HH_time2.append(self.c_HH * 1)
                    self.c_Notum_time2.append(self.c_Notum * 1)
                    self.c_NRF_time2.append(self.c_NRF * 1)
                    self.c_ERK_time2.append(self.c_ERK * 1)

                    self.Head_time2.append(self.Head*1)
                    self.Tail_time2.append(self.Tail*1)
                    self.Blast_time2.append(self.Blast*1)

                    # self.hdac_time2.append(self.hdac * 1)


                elif self.runtype == 'sim':

                    self.c_BC_sim_time.append(self.c_BC * 1)
                    self.c_WNT_sim_time.append(self.c_WNT * 1)
                    self.c_HH_sim_time.append(self.c_HH * 1)
                    self.c_Notum_sim_time.append(self.c_Notum * 1)
                    self.c_NRF_sim_time.append(self.c_NRF * 1)
                    self.c_ERK_sim_time.append(self.c_ERK * 1)

                    # self.hdac_sim_time.append(self.hdac * 1)

                    self.Head_sim_time.append(self.Head*1)
                    self.Tail_sim_time.append(self.Tail*1)
                    self.Blast_sim_time.append(self.Blast*1)


    def initialize(self,
                   knockdown= None,
                   run_time=48.0 * 3600,
                   run_time_step=60,
                   run_time_sample=100,
                   reset_clims = True,
                   ):

        if knockdown is None:
            knockdown = self.RNAi_defaults

        # set time parameters:
        self.tmin = 0.0
        self.tmax = run_time
        self.dt = run_time_step*self.x_scale
        self.tsamp = int(run_time_sample/self.x_scale)
        self.nt = int((self.tmax - self.tmin) / self.dt)
        self.time = np.linspace(self.tmin, self.tmax, self.nt)
        self.tsample = self.time[0:-1:self.tsamp]

        # scale the cell cluster by the given factor
        self.cells = self.scale_cells(self.x_scale)

        # save the cells object at this stage so that we can plot init later
        self.cells_i = copy.deepcopy(self.cells)

        self.clear_cache_init()
        self.runtype = 'init'
        self.run_loop(knockdown=knockdown)
        self.tsample_init = self.tsample

        # write the final molecule time arrays to the master dictionary:
        self.molecules_time['β-Cat'] = self.c_BC_time
        self.molecules_time['Erk'] = self.c_ERK_time
        self.molecules_time['Wnt'] = self.c_WNT_time
        self.molecules_time['Hh'] = self.c_HH_time
        self.molecules_time['NRF'] = self.c_NRF_time
        self.molecules_time['Notum'] = self.c_Notum_time
        self.molecules_time['Head'] = self.Head_time
        self.molecules_time['Tail'] = self.Tail_time

        if reset_clims:
            # Reset default clims to levels at the end of the initialization phase:
            # default plot legend scaling (can be modified)
            mol_clims = OrderedDict()

            mol_clims['β-Cat'] = [0, np.max(self.molecules_time['β-Cat'][-1])]
            mol_clims['Erk'] = [0, 1.0]
            mol_clims['Wnt'] = [0, np.max(self.molecules_time['Wnt'][-1])]
            mol_clims['Hh'] = [0, np.max(self.molecules_time['Hh'][-1])]
            mol_clims['NRF'] = [0, np.max(self.molecules_time['NRF'][-1])]
            mol_clims['Notum'] = [0, np.max(self.molecules_time['Notum'][-1])]
            mol_clims['NotumRNA'] = [0, 1.0]
            mol_clims['APC'] = [0, 1.0]
            mol_clims['cAMP'] = [0, 1.0]
            mol_clims['Head'] = [0.0, 1.0]
            mol_clims['Tail'] = [0.0, 1.0]

            self.default_clims = mol_clims



        if self.verbose:
            print("Successfully completed init of model!")

    def reinitialize(self,
                     knockdown=None,
                     run_time=48.0 * 3600,
                     run_time_step=10,
                     run_time_sample=100):

        if knockdown is None:
            knockdown = self.RNAi_defaults

        # set time parameters:
        self.tmin = 0.0
        self.tmax = run_time
        self.dt = run_time_step * self.x_scale
        self.tsamp = int(run_time_sample/self.x_scale)
        self.nt = int((self.tmax - self.tmin) / self.dt)
        self.time = np.linspace(self.tmin, self.tmax, self.nt)
        self.tsample = self.time[0:-1:self.tsamp]

        self.clear_cache_reinit()
        self.runtype = 'reinit'
        self.run_loop(knockdown=knockdown)
        self.tsample_reinit = self.tsample

        # write the final molecule time arrays to the master dictionary:
        self.molecules_time2['β-Cat'] = self.c_BC_time2
        self.molecules_time2['Erk'] = self.c_ERK_time2
        self.molecules_time2['Wnt'] = self.c_WNT_time2
        self.molecules_time2['Hh'] = self.c_HH_time2
        self.molecules_time2['NRF'] = self.c_NRF_time2
        self.molecules_time2['Notum'] = self.c_Notum_time2
        self.molecules_time2['Head'] = self.Head_time2
        self.molecules_time2['Tail'] = self.Tail_time2

        if self.verbose:
            print("Successfully completed reinit of model!")

    def simulate(self,
                 knockdown=None,
                 run_time=48.0 * 3600,
                 run_time_step=10,
                 run_time_sample=100,
                 reset_clims = False):

        if knockdown is None:
            knockdown = self.RNAi_defaults

        # set time parameters:
        self.tmin = 0.0
        self.tmax = run_time
        self.dt = run_time_step * self.x_scale
        self.tsamp = int(run_time_sample/self.x_scale)
        self.nt = int((self.tmax - self.tmin) / self.dt)
        self.time = np.linspace(self.tmin, self.tmax, self.nt)
        self.tsample = self.time[0:-1:self.tsamp]

        # Remove cells at user-specified cutlines and update all array indices:
        self.cut_cells()

        self.clear_cache_sim()
        self.runtype = 'sim'
        self.run_loop(knockdown=knockdown)

        self.tsample_sim = self.tsample

        # write the final molecule time arrays to the master dictionary:
        self.molecules_sim_time['β-Cat'] = self.c_BC_sim_time
        self.molecules_sim_time['Erk'] = self.c_ERK_sim_time
        self.molecules_sim_time['Wnt'] = self.c_WNT_sim_time
        self.molecules_sim_time['Hh'] = self.c_HH_sim_time
        self.molecules_sim_time['NRF'] = self.c_NRF_sim_time
        self.molecules_sim_time['Notum'] = self.c_Notum_sim_time
        self.molecules_sim_time['Head'] = self.Head_sim_time
        self.molecules_sim_time['Tail'] = self.Tail_sim_time

        if reset_clims:
            # Reset default clims to levels at the end of the initialization phase:
            # default plot legend scaling (can be modified)
            mol_clims = OrderedDict()

            mol_clims['β-Cat'] = [0, np.max(self.molecules_sim_time['β-Cat'][-1])]
            mol_clims['Erk'] = [0, 1.0]
            mol_clims['Wnt'] = [0, np.max(self.molecules_sim_time['Wnt'][-1])]
            mol_clims['Hh'] = [0, np.max(self.molecules_sim_time['Hh'][-1])]
            mol_clims['NRF'] = [0, np.max(self.molecules_sim_time['NRF'][-1])]
            mol_clims['Notum'] = [0, np.max(self.molecules_time['Notum'][-1])]
            mol_clims['NotumRNA'] = [0, 1.0]
            mol_clims['APC'] = [0, 1.0]
            mol_clims['cAMP'] = [0, 1.0]
            mol_clims['Head'] = [0.0, 1.0]
            mol_clims['Tail'] = [0.0, 1.0]

            self.default_clims = mol_clims

        if self.verbose:
            print("Successfully completed sim of model!")

    # Saving and loading functions ----------------------------
    def save_sim(self, fname):

        pickles.save(self, filename=fname, is_overwritable=True)

    def load_sim(fname):

        master = pickles.load(fname)

        return master

    # Workup functions-----------------------------------------
    def work_data(self, ti =-1, run_type = 'init', substance = 'Erk', ref_data = None):

        if ref_data is not None:
            # Plot an init:
            if run_type == 'init':
                ref = ref_data[0][substance][ti]
                carray = self.molecules_time[substance][ti]

            elif run_type == 'sim':
                ref = ref_data[1][substance][ti]
                carray = self.molecules_sim_time[substance][ti]

            diffi = ((carray - ref)/ref)*100 # percent difference initial to final at each data point
            diff = np.sum(diffi)*(1/self.cdl) # average percent difference whole model

        else:
            diff = None

        return diff

    # Markov processing functions------------------------------
    @abstractmethod
    def get_tops(self, cinds):
        pass

    def process_markov(self, head_i, tail_i):
        """
        Post-processing of the Markov model to return heteromorphoses probabilities for cut fragments
        :param head_i: user-specified framgent representing head
        :param tail_i: user-specified fragment representing tail

        """


        head_frag = head_i
        tail_frag = tail_i

        frag_probs = OrderedDict()
        for fragn in self.fragments.keys():
            frag_probs[fragn] = OrderedDict()

        for fragn, wounds_arr in self.frags_and_wounds.items():

            wound_num = len(wounds_arr)

            if wound_num == 1 and fragn in head_frag:

                frag_probs[fragn]['pHa'] = 1.0
                frag_probs[fragn]['pTa'] = 0.0
                frag_probs[fragn]['pBa'] = 0.0

                pHb, pTb, pBb = self.get_tops(wounds_arr[0])

                frag_probs[fragn]['pHb'] = pHb
                frag_probs[fragn]['pTb'] = pTb
                frag_probs[fragn]['pBb'] = pBb

            elif wound_num == 1 and fragn in tail_frag:

                frag_probs[fragn]['pHa'] = 0.0
                frag_probs[fragn]['pTa'] = 1.0
                frag_probs[fragn]['pBa'] = 0.0

                pHb, pTb, pBb = self.get_tops(wounds_arr[0])

                frag_probs[fragn]['pHb'] = pHb
                frag_probs[fragn]['pTb'] = pTb
                frag_probs[fragn]['pBb'] = pBb

            elif wound_num == 2:

                pHa, pTa, pBa = self.get_tops(wounds_arr[0])
                pHb, pTb, pBb = self.get_tops(wounds_arr[1])

                frag_probs[fragn]['pHa'] = pHa
                frag_probs[fragn]['pTa'] = pTa
                frag_probs[fragn]['pBa'] = pBa

                frag_probs[fragn]['pHb'] = pHb
                frag_probs[fragn]['pTb'] = pTb
                frag_probs[fragn]['pBb'] = pBb

        morph_probs = OrderedDict()
        for fragn in self.fragments.keys():
            morph_probs[fragn] = OrderedDict()

        for fragn, prob_dict in frag_probs.items():

            check_len = len(prob_dict.values())

            if check_len == 6:
                pHa = prob_dict['pHa']
                pTa = prob_dict['pTa']
                pBa = prob_dict['pBa']

                pHb = prob_dict['pHb']
                pTb = prob_dict['pTb']
                pBb = prob_dict['pBb']

                p2T = pTa * pTb
                p0H = (pTa * pBb + pTb * pBa)
                p1H = (pHa * pTb + pHb * pTa)
                p0T = (pHa * pBb + pHb * pBa)
                p2H = pHa * pHb

                morph_probs[fragn]['2T'] = p2T
                morph_probs[fragn]['0H'] = p0H
                morph_probs[fragn]['1H'] = p1H
                morph_probs[fragn]['0T'] = p0T
                morph_probs[fragn]['2H'] = p2H

                pBlob = 1.0 - p2T - p0H - p1H - p0T - p2H # Blobs have neither head nor tail...
                morph_probs[fragn]['00'] = pBlob

        # probability of head/tail/fail outcomes at each wound:
        self.frag_probs = frag_probs

        # probability of heteromorphoses in each fragment:
        self.morph_probs = morph_probs

    def heteromorph_table(self, transpose = False):
        """
        Produces a data table of fragments in rows and heteromorph probabilities in columns
         (or reverse, if transpose is True), which is suitable for adding to a plot.

        """
        morph_data = []

        col_tags = ['2T', '0H', '1H', '0T', '2H', '00']
        row_tags = []

        for frag_n, hmorphs, in self.morph_probs.items():

            p2T = np.round(hmorphs['2T'],2)
            p0H = np.round(hmorphs['0H'],2)
            p1H = np.round(hmorphs['1H'],2)
            p0T = np.round(hmorphs['0T'],2)
            p2H = np.round(hmorphs['2H'],2)
            p00 = np.round(hmorphs['00'],2)

            row_data = [p2T, p0H, p1H, p0T, p2H, p00]

            morph_data.append(row_data)
            row_tags.append('Frag ' + str(frag_n))

        morph_data = np.asarray(morph_data)

        if transpose is True: # flip everything around
            morph_data = morph_data.T
            # reassign column and row tags
            col_tags = row_tags
            row_tags = ['2T', '0H', '1H', '0T', '2H', '00']



        return morph_data, col_tags, row_tags



    # Plotting functions---------------------------------------

    @abstractmethod
    def init_plots(self):

        pass

    @abstractmethod
    def triplot(self, ti,
                fname = 'Triplot_', dirsave = None, reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (12, 12), clims = None, autoscale = True,
                      ref_data = None, extra_text = None, txt_x = 0.05, txt_y = 0.92):

        pass

    @abstractmethod
    def biplot(self, ti, plot_type = 'init', fname = 'Biplot_', dirsave = None, reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (12, 12), clims = None, autoscale = True,
               ref_data=None, extra_text = None, txt_x = 0.05, txt_y = 0.92):

        pass

    @abstractmethod
    def plot(self, ti, ctag, plot_type='init', dirsave = 'Plot', reso = 150, linew = 3.0,
                cmaps=None, fontsize=16.0, fsize=(10, 6), clims = None, autoscale = True):

        pass

    @abstractmethod
    def animate_triplot(self, ani_type = 'init', dirsave = None, reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (12, 12), clims = None, autoscale = True,
                        ref_data=None, extra_text = None):

        pass


    @abstractmethod
    def animate_biplot(self, ani_type='init', dirsave = None, reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (12, 12), clims = None, autoscale = True,
                       ref_data=None, extra_text = None):
        pass

    @abstractmethod
    def animate_plot(self, ctag, ani_type='init', dirsave = 'PlotAni', reso = 150, linew = 3.0,
                cmaps=None, fontsize=16.0, fsize=(10, 6), clims = None, autoscale = True):

        pass

    @abstractmethod
    def hexplot(self, ti, plot_type = 'init',  fname = 'Hexplot_', dirsave = None, reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (16, 12), clims = None, autoscale = True,
                      ref_data = None, extra_text = None, txt_x = 0.05, txt_y = 0.92):

        pass


    @abstractmethod
    def markovplot(self, ti, plot_type = 'init',  fname = 'Markov_', dirsave = None, reso = 150, linew = 3.0,
                      cmaps = None, fontsize = 16.0, fsize = (12, 12), clims = None, autoscale = True,
                      ref_data = None, extra_text = None, txt_x = 0.05, txt_y = 0.92):

        pass

    @abstractmethod
    def plot_frags(self, show_plot = False, save_plot = True, fsize=(12, 12),
                   reso = 150, group_colors = None, dir_save = None):
        pass




