#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright 2018-2019 by Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''

Harness for running automated model searches and sensitivity analyses for the 1D and 2D simulators of the
PLIMBO module.

'''

import numpy as np
from collections import OrderedDict
import os
import os.path
# --- SOVEREIGN AUTOMATED CORRECTION ---
try:
    from betse.util.serial.pickle import pickles  # Modern BETSE path
except ImportError:
    from betse.lib.pickle import pickles         # Legacy fallbackfrom betse.lib.pickle import pickles
from betse.util.path import dirs, pathnames

from plimbo.sim1D import PlanariaGRN1D
from plimbo.sim2D import PlanariaGRN2D
from plimbo.auto_params import ParamsManager


class ModelHarness(object):

    def __init__(self, config_filename, paramo = None, xscale=1.0, harness_type = '1D', plot_frags = True,
                 verbose = False, new_mesh=False, savedir = 'ModelSearch', head_frags = None, tail_frags = None):
        """
        For the model formalism using the default bitmaps for 2D planaria simulations provided,
        the 'head_frag' of a 1H model is specified as [0], while the tail_frag is specified as [4] for the 4-cut model (
        the head fragment index will always be zero, while the tail_fragment will equal the number of cuts). This
        is also the case for 1D simulations. To specify a 2H model, supply the default tail fragment index to
          the head_frags array, for example: [0,4].

        :param config_filename:
        :param paramo:
        :param xscale:
        :param harness_type:
        :param plot_frags:
        :param verbose:
        :param new_mesh:
        :param savedir:
        :param head_frags:
        :param tail_frags:
        """

        self.xscale = xscale

        self.paramo = paramo

        self.verbose = verbose

        self.new_mesh = new_mesh

        self.savedir = savedir

        self.config_fn = config_filename

        # specify fragments that are heads or tails for the Markov simulation:
        if head_frags is None:
            self.head_frags = [0]

        else:
            self.head_frags = head_frags

        if tail_frags is None:
            self.tail_frags = [4]

        else:
            self.tail_frags = tail_frags

        # Create a simulator object:
        if harness_type == '1D':
            self.model = PlanariaGRN1D(config_filename, self.paramo, xscale=self.xscale, verbose=False, new_mesh=False)

        if harness_type == '2D':
            self.model = PlanariaGRN2D(config_filename, self.paramo, xscale=self.xscale, verbose=False, new_mesh=False)

        if self.paramo is None:
            self.paramo = self.model.pdict

        # save the harness type so we know what we're working with:
        self.harness_type = harness_type

        # Create a directory to store results:
        self.savepath = pathnames.join_and_canonicalize(self.model.p.conf_dirname, savedir)

        os.makedirs(self.savepath, exist_ok=True)

        # Create a dictionary to save sub-folder paths:
        self.subfolders_dict = OrderedDict()

        # Generate a parameters manager object for the harness:
        self.pm = ParamsManager(self.paramo)

        self.has_autoparams = False

        # initialize outputs array to null:
        self.outputs = []

        # save the model's concentration tags:
        self.conctags = self.model.conc_tags

        # reference data set to null
        self.ref_data = None

        # extra information to write on plots:
        self.plot_info_msg = None


        # default RNAi testing sequence vector:
        self.RNAi_vect_default = [
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 0.25,
             'dynein': 1, 'kinesin': 1, 'ptc': 1},
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 5.0,
             'dynein': 1, 'kinesin': 1, 'ptc': 1},
            {'bc': 0.0, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 1, 'ptc': 1},
            {'bc': 1, 'erk': 0.0, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 1, 'ptc': 1},
            {'bc': 1, 'erk': 1, 'apc': 0.0, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 1, 'ptc': 1},
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 0.0, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 1, 'ptc': 1},
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 1, 'ptc': 0.0},
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 0.0, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 1, 'ptc': 1},
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 0.0, 'camp': 1,
             'dynein': 1, 'kinesin': 1, 'ptc': 1},
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 0.0, 'kinesin': 1, 'ptc': 1},
            {'bc': 1, 'erk': 1, 'apc': 1, 'notum': 1, 'wnt': 1, 'hh': 1, 'camp': 1,
             'dynein': 1, 'kinesin': 0.0, 'ptc': 1},

        ]

        self.RNAi_tags_default = ['cAMP_Down', 'cAMP_Up', 'RNAi_BC', 'RNAi_ERK', 'RNAi_APC',
                                  'RNAi_Notum', 'RNAi_Ptc', 'RNAi_WNT', 'RNAi_HH', 'Dynein', 'Kinesin']

        self.xscales_default = [0.5, 1.5, 3.0]




    def run_sensitivity(self, factor = 0.1, verbose=True, run_time_init = 36000.0,
                        run_time_sim = 36000.0, run_time_step = 60, run_time_sample = 50,
                        reset_clims = True, animate = False, plot = True, plot_type = ['Triplot'],
                        save_dir = 'Sensitivity1', ani_type = 'Triplot', axisoff = False,
                        save_all = False, data_output = True, reference = ['Head','Tail'], fsize=[(6,8)],
                        clims = None, paramo_units = None, run_type = 'init'):

        if paramo_units is None:

            # units listing for the parameters in the model:
            self.params_units = OrderedDict({

                # Beta cat parameters
                'r_bc': 'nM/s',
                'd_bc': '1/s',
                'd_bc_deg': '1/s',
                'K_bc_apc': 'nM',
                'n_bc_apc': '',
                'K_bc_camp': 'nM',
                'n_bc_camp': ' ',
                'D_bc': 'm^2/s',

                # ERK parameters
                'K_erk_bc': 'nM',
                'n_erk_bc': ' ',

                # APC parameters
                'K_apc_wnt': 'nM',
                'n_apc_wnt': ' ',

                # Hedgehog parameters:
                'r_hh': 'nM/s',
                'd_hh': '1/s',
                'D_hh': 'm^2/s',
                'u_hh': 'm/s',

                # Wnt parameters
                'r_wnt': 'nM/s',
                'd_wnt': '1/s',
                'K_wnt_notum': 'nM',
                'n_wnt_notum': ' ',
                'D_wnt': 'm^2/s',
                'd_wnt_deg_notum': '1/s',
                'd_wnt_deg_ptc': '1/s',
                'K_wnt_hh': 'nM',
                'n_wnt_hh': ' ',

                # NRF parameters
                'r_nrf': 'nM/s',
                'd_nrf': '1/s',
                'K_nrf_bc': 'nM/m^3',
                'n_nrf_bc': ' ',
                'D_nrf': 'm^2/s',
                'u_nrf': 'm/s',

                # Notum parameters
                'r_notum': 'nM/s',
                'd_notum': '1/s',
                'K_notum_nrf': 'nM',
                'n_notum_nrf': ' ',
                'D_notum': 'm^2/s',

                # Markov model parameters:
                'C1': 'nM',  # ERK constant to modulate head formation
                'K1': 'nM',

                'C2': 'nM',  # Beta-catenin concentration to modulate tail formation
                'K2': 'nM',

                'Beta_B': '1/s',  # head/tail tissue decay time constant

                'hdac_to': 's',  # time at which hdac stops growing
                'hdac_ts': 's',  # time period over which hdac stops growing

                'nd_min': '',
                'nd_max': ''

            })

        else:
            self.params_units = paramo_units

        # general saving directory for this procedure:
        self.savedir_sensitivity = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_sensitivity, exist_ok=True)

        self.subfolders_dict['sensitivity'] = self.savedir_sensitivity

        # Create datatags for the harness to save data series to:
        self.datatags = []

        self.datatags.append('base')

        self.pm.create_sensitivity_matrix(factor=factor)  # Generate sensitivity matrix from default parameters
        self.has_autoparams = True

        self.outputs = []  # Storage array for all data created in each itteration of the model
        self.heteromorphoses = [] # Storage array for heteromorph probabilities

        for ii, params_list in enumerate(self.pm.params_M):  # Step through the parameters matrix

            try:

                data_dict_inits = OrderedDict()  # Storage array for full molecules array created in each model init
                data_dict_sims = OrderedDict()  # Storage array for full molecules array created in each model sim
                data_dict_prob = OrderedDict()  # storage of fragment probabilities for each itteration

                if verbose is True:
                    print('Run ', ii + 1, " of ", self.pm.N_runs)

                # convert the array to a dictionary:
                run_params = OrderedDict(zip(self.pm.param_labels, params_list))

                # create a model using the specific parameters from the params manager for this run:
                self.model.model_init(self.config_fn, run_params, xscale=self.xscale,
                                    verbose=self.verbose, new_mesh=self.new_mesh)

                # Run initialization of full model:
                self.model.initialize(knockdown= None,
                                       run_time=run_time_init,
                                       run_time_step=run_time_step,
                                       run_time_sample=run_time_sample,
                                       reset_clims = reset_clims)

                if data_output:
                    self.output_clims(ri=ii, save_dir=self.savedir_sensitivity, run_type='init')

                self.model.simulate(knockdown= None,
                                       run_time=run_time_sim,
                                       run_time_step=run_time_step,
                                       run_time_sample=run_time_sample,
                                       reset_clims = reset_clims)

                if data_output:
                    self.output_clims(ri=ii, save_dir=self.savedir_sensitivity, run_type='sim')

                # if we're on the first timestep, set it as the reference data set:
                if ii == 0:
                    self.ref_data = [self.model.molecules_time.copy(), self.model.molecules_sim_time.copy()]

                # if we're past the first timesep, prepare messages for the plots about how params have changed:
                if ii > 0:
                    self.write_plot_msg(ii)

                data_dict_inits['base'] = self.model.molecules_time.copy()
                data_dict_sims['base'] = self.model.molecules_sim_time.copy()

                self.outputs.append([data_dict_inits, data_dict_sims])

                self.model.process_markov(self.head_frags, self.tail_frags)
                data_dict_prob['base'] = self.model.morph_probs.copy()

                self.heteromorphoses.append(data_dict_prob)

                if plot:

                    self.plot_single('base', ii, harness_type='sensitivity', plot_type=plot_type, axisoff = axisoff,
                                     output_type='init', ref_data=self.ref_data[0], extra_text = self.plot_info_msg,
                                     fsize=fsize, clims=clims)

                    self.plot_single('base', ii, harness_type='sensitivity', plot_type=plot_type, axisoff = axisoff,
                                     output_type='sim', ref_data=self.ref_data[1], extra_text = self.plot_info_msg,
                                     fsize=fsize, clims=clims)

                    self.output_heteromorphs_single('base', ii, save_dir=self.savedir_sensitivity)

                if animate:
                    self.ani_single('base', ii, harness_type='sensitivity', ani_type=ani_type, axisoff = axisoff,
                                    output_type='init', ref_data=self.ref_data[0], extra_text = self.plot_info_msg,
                                    fsize=fsize, clims=clims)
                    self.ani_single('base', ii, harness_type='sensitivity', ani_type=ani_type, axisoff = axisoff,
                                    output_type='sim', ref_data=self.ref_data[1], extra_text = self.plot_info_msg,
                                    fsize=fsize, clims=clims)

                if verbose is True:
                    print('----------------')


            except:
                print('***************************************************')
                print("Run", ii + 1, "has become unstable and been terminated.")
                print('***************************************************')

        if data_output:
            self.output_delta_table(substance=reference, run_type=run_type, save_dir=self.savedir_sensitivity)
            self.output_summary_table(substance=reference, run_type=run_type, save_dir=self.savedir_sensitivity)
            self.output_heteromorphs(save_dir=self.savedir_sensitivity)
            self.model.plot_frags(dir_save=self.savedir_sensitivity, fsize=fsize[0])

        if save_all:
            fsave = os.path.join(self.savedir_sensitivity, "Master.gz")
            self.save(fsave)


    def run_searchRNAi(self, RNAi_series = None, RNAi_names = None, factor = 0.8, levels = 1, search_style = 'log',
                        verbose=True, run_time_reinit=0.0, run_time_init=36000.0, run_time_sim=36000.0, save_all = False,
                       run_time_step=60, run_time_sample=50, reset_clims=True, plot=True, ani_type = 'Triplot',
                       animate=False, save_dir='SearchRNAi1', free_params = None, plot_type = ['Triplot'],
                       data_output = True, axisoff = False, fsize=[(6,8)], clims = None, up_only = False):

        if RNAi_series is None or RNAi_names is None:

            RNAi_series = self.RNAi_vect_default
            RNAi_names = self.RNAi_tags_default

        # general saving directory for this procedure:
        self.savedir_searchRNAi = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_searchRNAi, exist_ok=True)

        self.subfolders_dict['searchRNAi'] = self.savedir_searchRNAi

        # Create datatags for the harness to save data series to:
        self.datatags = []

        self.datatags.append('base')

        if len(RNAi_series): # if there's anything in the RNAi series:
            # add in new datatags for RNAi_series
            for rnai_n in RNAi_names:
                self.datatags.append(rnai_n)

        # Create the parameters matrix:
        self.pm.create_search_matrix(factor=factor, levels=levels, style=search_style,
                                     free_params=free_params, up_only = up_only)
        self.has_autoparams = True

        self.outputs = []  # Storage array for all last timestep outputs
        self.heteromorphoses = [] # Storage array for heteromorph probabilities

        for ii, params_list in enumerate(self.pm.params_M):  # Step through the parameters matrix

            try:

                if verbose is True:
                    print('Run ', ii + 1, " of ", self.pm.N_runs)

                data_dict_inits = OrderedDict()  # Storage array for full molecules array created in each model init
                data_dict_sims = OrderedDict()  # Storage array for full molecules array created in each model sim
                data_dict_prob = OrderedDict()  # storage of fragment probabilities for each itteration

                # convert the array to a dictionary:
                run_params = OrderedDict(zip(self.pm.param_labels, params_list))

                # create a model using the specific parameters from the params manager for this run:
                self.model.model_init(self.config_fn, run_params, xscale=self.xscale,
                                    verbose=self.verbose, new_mesh=self.new_mesh)

                # Run initialization of full model:
                self.model.initialize(knockdown= None,
                                       run_time=run_time_init,
                                       run_time_step=run_time_step,
                                       run_time_sample=run_time_sample,
                                       reset_clims = reset_clims)

                if data_output:
                    self.output_clims(ri=ii, save_dir=self.savedir_searchRNAi, run_type='init')

                self.model.simulate(knockdown= None,
                                       run_time=run_time_sim,
                                       run_time_step=run_time_step,
                                       run_time_sample=run_time_sample,
                                       reset_clims = reset_clims)

                if data_output:
                    self.output_clims(ri=ii, save_dir=self.savedir_searchRNAi, run_type='sim')

                # if we're on the first timestep, set it as the reference data set:
                if ii == 0:
                    self.ref_data = [self.model.molecules_time.copy(), self.model.molecules_sim_time.copy()]

                # if we're past the first timesep, prepare messages for the plots about how params have changed:
                if ii > 0:
                    self.write_plot_msg(ii)


                data_dict_inits['base'] = self.model.molecules_time.copy()
                data_dict_sims['base'] = self.model.molecules_sim_time.copy()

                self.model.process_markov(self.head_frags, self.tail_frags)
                data_dict_prob['base'] = self.model.morph_probs.copy()

                if plot:
                    self.plot_single('base', ii, harness_type='searchRNAi', plot_type=plot_type, axisoff = axisoff,
                                     output_type='init', ref_data = self.ref_data[0], extra_text = self.plot_info_msg,
                                     fsize=fsize, clims=clims)
                    self.plot_single('base', ii, harness_type='searchRNAi', plot_type=plot_type, axisoff = axisoff,
                                     output_type='sim', ref_data = self.ref_data[1], extra_text = self.plot_info_msg,
                                     fsize=fsize, clims=clims)

                    self.output_heteromorphs_single('base', ii, save_dir=self.savedir_searchRNAi)



                if animate:
                    self.ani_single('base', ii, harness_type='searchRNAi', ani_type=ani_type, axisoff = axisoff,
                                    output_type='init', ref_data = self.ref_data[0], extra_text = self.plot_info_msg,
                                    fsize=fsize, clims=clims)
                    self.ani_single('base', ii, harness_type='searchRNAi', ani_type=ani_type, axisoff = axisoff,
                                    output_type='sim', ref_data = self.ref_data[1], extra_text = self.plot_info_msg,
                                    fsize=fsize, clims=clims)

                if verbose is True:
                    print('----------------')

                if len(RNAi_series): # if there's anything in the RNAi test series:

                    for rnai_s, rnai_n in zip(RNAi_series, RNAi_names):

                        if verbose is True:
                            print('Runing RNAi Sequence ', rnai_n)

                        # Reinitialize the model again:
                        self.model.model_init(self.config_fn, run_params, xscale=self.xscale,
                                              verbose=self.verbose, new_mesh=self.new_mesh)

                        # Run initialization phase of full model:
                        self.model.initialize(knockdown=None,
                                              run_time=run_time_init,
                                              run_time_step=run_time_step,
                                              run_time_sample=run_time_sample,
                                              reset_clims=reset_clims)

                        if run_time_reinit > 0.0:  # if there is a reinitialization phase (RNAi applied, no cutting)
                            self.model.reinitialize(knockdown=rnai_s,
                                                    run_time=run_time_reinit,
                                                    run_time_step=run_time_step,
                                                    run_time_sample=run_time_sample
                                                    )
                        # Run the simulation with RNAi intervention applied:
                        self.model.simulate(knockdown=rnai_s,
                                            run_time=run_time_sim,
                                            run_time_step=run_time_step,
                                            run_time_sample=run_time_sample,
                                            reset_clims=False)

                        # Save whole molecules master arrays to their respective data dictionaries:
                        data_dict_inits[rnai_n] = self.model.molecules_time.copy()
                        data_dict_sims[rnai_n] = self.model.molecules_sim_time.copy()

                        self.model.process_markov(self.head_frags, self.tail_frags)
                        data_dict_prob[rnai_n] = self.model.morph_probs.copy()

                        if plot:
                            self.plot_single(rnai_n, ii, harness_type='searchRNAi', plot_type=plot_type,
                                             axisoff = axisoff,
                                             output_type='sim', ref_data = self.ref_data[1],
                                             extra_text = self.plot_info_msg,
                                             fsize=fsize,
                                             clims=clims)

                            self.output_heteromorphs_single(rnai_n, ii, save_dir=self.savedir_searchRNAi)

                        if animate:
                            self.ani_single(rnai_n, ii, harness_type='searchRNAi', ani_type=ani_type, axisoff = axisoff,
                                            output_type='sim', ref_data = self.ref_data[1],
                                            extra_text = self.plot_info_msg, fsize=fsize, clims=clims)

                        if verbose is True:
                            print('----------------')

                    self.outputs.append([data_dict_inits, data_dict_sims])
                    self.heteromorphoses.append(data_dict_prob)

            except:

                print('***************************************************')
                print("Run", ii +1, "has become unstable and been terminated.")
                print('***************************************************')


        if data_output:
            self.output_delta_table(substance=['Head', 'Tail'], run_type='init', save_dir=self.savedir_searchRNAi)
            self.output_summary_table(substance=['Head','Tail'], run_type='init', save_dir=self.savedir_searchRNAi)
            self.output_heteromorphs(save_dir=self.savedir_searchRNAi)
            self.model.plot_frags(dir_save=self.savedir_searchRNAi, fsize=fsize[0])

        if save_all:
            fsave = os.path.join(self.savedir_searchRNAi, "Master.gz")
            self.save(fsave)

    def run_scaleRNAi(self, xscales = None, RNAi_series = None, RNAi_names = None, verbose=True,
                       run_time_reinit=0.0, run_time_init=36000.0, run_time_sim=36000.0,
                       run_time_step=60, run_time_sample=50, reset_clims=True, plot=True,
                       animate=False, save_dir='scaleRNAi1', plot_type = ['Triplot'], save_all = False,
                       ani_type = 'Triplot', data_output = True, axisoff = False, fsize=[(6,8)], clims = None
                       ):

        if RNAi_series is None or RNAi_names is None:

            RNAi_series = self.RNAi_vect_default
            RNAi_names = self.RNAi_tags_default

        if xscales is None:
            xscales = self.xscales_default

        # general saving directory for this procedure:
        self.savedir_scaleRNAi = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_scaleRNAi, exist_ok=True)

        self.subfolders_dict['scaleRNAi'] = self.savedir_scaleRNAi

        # Create datatags for the harness to save data series to:
        self.datatags = []

        self.datatags.append('base')

        if len(RNAi_series): # if there's anything in the series:
            # add in new datatags for RNAi_series
            for rnai_n in RNAi_names:
                self.datatags.append(rnai_n)

        self.outputs = []  # Storage array for all last timestep outputs
        self.heteromorphoses = [] # Storage array for heteromorph probabilities

        for ii, xxs in enumerate(xscales):  # Step through the x-scaling factors

            try:

                if verbose is True:
                    print('Run ', ii + 1, " of ", len(xscales))

                data_dict_inits = OrderedDict()  # Storage array for full molecules array created in each model init
                data_dict_sims = OrderedDict()  # Storage array for full molecules array created in each model sim
                data_dict_prob = OrderedDict()  # storage of fragment probabilities for each itteration

                # create a model using the specific parameters from the params manager for this run at this scale:
                self.model.model_init(self.config_fn, self.paramo, xscale=xxs,
                                      verbose=self.verbose, new_mesh=self.new_mesh)

                # Run initialization of full model:
                self.model.initialize(knockdown=None,
                                      run_time=run_time_init,
                                      run_time_step=run_time_step,
                                      run_time_sample=run_time_sample,
                                      reset_clims=reset_clims)

                if data_output:
                    self.output_clims(ri=ii, save_dir=self.savedir_scaleRNAi, run_type='init')

                self.model.simulate(knockdown=None,
                                    run_time=run_time_sim,
                                    run_time_step=run_time_step,
                                    run_time_sample=run_time_sample,
                                    reset_clims=reset_clims)

                if data_output:
                    self.output_clims(ri=ii, save_dir=self.savedir_scaleRNAi, run_type='sim')

                data_dict_inits['base'] = self.model.molecules_time.copy()
                data_dict_sims['base'] = self.model.molecules_sim_time.copy()

                # Set reference data set to the main model curves::
                self.ref_data = [self.model.molecules_time.copy(), self.model.molecules_sim_time.copy()]

                self.model.process_markov(self.head_frags, self.tail_frags)
                data_dict_prob['base'] = self.model.morph_probs.copy()

                if plot:
                    self.plot_single('base', ii, harness_type='scaleRNAi', plot_type=plot_type, axisoff = axisoff,
                                     output_type='init', extra_text = self.plot_info_msg, fsize=fsize, clims=clims)
                    self.plot_single('base', ii, harness_type='scaleRNAi', plot_type=plot_type, axisoff = axisoff,
                                     output_type='sim', extra_text = self.plot_info_msg, fsize=fsize, clims=clims)
                    self.output_heteromorphs_single('base', ii, save_dir=self.savedir_scaleRNAi)

                if animate:
                    self.ani_single('base', ii, harness_type='scaleRNAi', ani_type=ani_type, axisoff = axisoff,
                                    output_type='init', extra_text = self.plot_info_msg, fsize=fsize, clims=clims)
                    self.ani_single('base', ii, harness_type='scaleRNAi', ani_type=ani_type, axisoff = axisoff,
                                    output_type='sim', extra_text = self.plot_info_msg, fsize=fsize, clims=clims)

                if verbose is True:
                    print('----------------')


                if len(RNAi_series): # if there's anything in the series, run it:

                    for rnai_s, rnai_n in zip(RNAi_series, RNAi_names):

                        if verbose is True:
                            print('Runing RNAi Sequence ', rnai_n)

                        # Reinitialize the model again:
                        self.model.model_init(self.config_fn, self.paramo, xscale=xxs,
                                              verbose=self.verbose, new_mesh=self.new_mesh)

                        # Run initialization phase of full model:
                        self.model.initialize(knockdown=None,
                                              run_time=run_time_init,
                                              run_time_step=run_time_step,
                                              run_time_sample=run_time_sample,
                                              reset_clims=reset_clims)

                        if run_time_reinit > 0.0:  # if there is a reinitialization phase (RNAi applied, no cutting)
                            self.model.reinitialize(knockdown=rnai_s,
                                                    run_time=run_time_reinit,
                                                    run_time_step=run_time_step,
                                                    run_time_sample=run_time_sample
                                                    )
                        # Run the simulation with RNAi intervention applied:
                        self.model.simulate(knockdown=rnai_s,
                                            run_time=run_time_sim,
                                            run_time_step=run_time_step,
                                            run_time_sample=run_time_sample,
                                            reset_clims=False)

                        # Save whole molecules master arrays to their respective data dictionaries:
                        data_dict_inits[rnai_n] = self.model.molecules_time.copy()
                        data_dict_sims[rnai_n] = self.model.molecules_sim_time.copy()

                        self.model.process_markov(self.head_frags, self.tail_frags)
                        data_dict_prob[rnai_n] = self.model.morph_probs.copy()

                        if plot:
                            self.plot_single(rnai_n, ii, harness_type='scaleRNAi', plot_type=plot_type, axisoff = axisoff,
                                             output_type='sim', ref_data=self.ref_data[1], fsize=fsize, clims=clims)

                        if animate:
                            self.ani_single(rnai_n, ii, harness_type='scaleRNAi', ani_type=ani_type, axisoff = axisoff,
                                            output_type='sim', ref_data=self.ref_data[1], fsize=fsize, clims=clims)

                        if verbose is True:
                            print('----------------')

                    self.outputs.append([data_dict_inits, data_dict_sims])
                    self.heteromorphoses.append(data_dict_prob)

            except:

                print('***************************************************')
                print("Run", ii +1 , "has become unstable and been terminated.")
                print('***************************************************')

        if data_output:
            self.output_heteromorphs(save_dir=self.savedir_scaleRNAi)
            self.model.plot_frags(dir_save=self.savedir_scaleRNAi, fsize=fsize[0])


        if save_all:
            fsave = os.path.join(self.savedir_scaleRNAi, "Master.gz")
            self.save(fsave)

    def run_simRNAi(self, RNAi_series = None, RNAi_names = None, verbose=True,
                    run_time_init = 36000.0, run_time_sim = 36000.0, run_time_step = 60,
                    run_time_sample = 50, run_time_reinit = 12, reset_clims = True,
                    plot_type = ['Triplot'], ani_type = 'Triplot', animate = False, save_all = False,
                    plot = True, save_dir = 'SimRNAi_1', data_output = True, axisoff = False, fsize=[(6,8)],
                    clims = None):

        if RNAi_series is None or RNAi_names is None:

            RNAi_series = self.RNAi_vect_default
            RNAi_names = self.RNAi_tags_default

        # general saving directory for this procedure:
        self.savedir_simRNAi = os.path.join(self.savepath, save_dir)
        os.makedirs(self.savedir_simRNAi, exist_ok=True)

        self.subfolders_dict['simRNAi'] = self.savedir_simRNAi


        # Create datatags for the harness to save data series to:
        self.datatags = []

        self.datatags.append('base')

        if len(RNAi_series): # if there's anything in the series, run it:

            # add in new datatags for RNAi_series
            for rnai_n in RNAi_names:
                data_tag = rnai_n
                self.datatags.append(data_tag)

        self.outputs = []  # Storage array for outputs of each model itteration
        self.heteromorphoses = [] # Storage array for heteromorph probabilities

        data_dict_inits = OrderedDict() # storage of inits for each molecules of a model itterantion
        data_dict_sims = OrderedDict() # storage of sims for each molecules of a model itterantion
        data_dict_prob = OrderedDict()  # storage of fragment probabilities for each itteration

        # create a model using the specific parameters from the params manager for this run:
        self.model.model_init(self.config_fn, self.paramo, xscale=self.xscale,
                              verbose=self.verbose, new_mesh=self.new_mesh)

        # Run initialization of full model:
        self.model.initialize(knockdown=None,
                              run_time=run_time_init,
                              run_time_step=run_time_step,
                              run_time_sample=run_time_sample,
                              reset_clims=reset_clims)

        if data_output:
            self.output_clims(save_dir=self.savedir_simRNAi, run_type='init')

        # Run simulation of full model:
        self.model.simulate(knockdown=None,
                              run_time=run_time_sim,
                              run_time_step=run_time_step,
                              run_time_sample=run_time_sample,
                              reset_clims=reset_clims)

        if data_output:
            self.output_clims(save_dir=self.savedir_simRNAi, run_type='sim')

        # Save whole molecules master arrays to their respective data dictionaries:
        data_dict_inits['base'] = self.model.molecules_time.copy()
        data_dict_sims['base'] = self.model.molecules_sim_time.copy()

        # Set reference data set to the main model curves::
        self.ref_data = [self.model.molecules_time.copy(), self.model.molecules_sim_time.copy()]

        self.model.process_markov(self.head_frags, self.tail_frags)
        data_dict_prob['base'] = self.model.morph_probs.copy()


        if plot:
            self.plot_single('base', 0, harness_type='simRNAi', plot_type=plot_type, axisoff = axisoff,
                             output_type='init', extra_text = self.plot_info_msg, fsize=fsize, clims = clims)
            self.plot_single('base', 0, harness_type='simRNAi', plot_type=plot_type, axisoff = axisoff,
                             output_type='sim', extra_text = self.plot_info_msg, fsize=fsize, clims = clims)
            self.output_heteromorphs_single('base', 0, save_dir=self.savedir_simRNAi)

        if animate:
            self.ani_single('base', 0, harness_type='simRNAi', ani_type=ani_type, axisoff = axisoff,
                            output_type='init', extra_text = self.plot_info_msg, fsize=fsize, clims = clims)
            self.ani_single('base', 0, harness_type='simRNAi', ani_type=ani_type, axisoff = axisoff,
                            output_type='sim', extra_text = self.plot_info_msg, fsize=fsize, clims = clims)

        if verbose is True:
            print('----------------')

        if len(RNAi_series): # if there's anything in the RNAi test series, run it:

            for rnai_s, rnai_n in zip(RNAi_series, RNAi_names):

                if verbose is True:
                    print('Runing RNAi Sequence ', rnai_n)

                # Reinitialize the model again:
                self.model.model_init(self.config_fn, self.paramo, xscale=self.xscale,
                                      verbose=self.verbose, new_mesh=self.new_mesh)

                # Run initialization phase of full model:
                self.model.initialize(knockdown=None,
                                      run_time=run_time_init,
                                      run_time_step=run_time_step,
                                      run_time_sample=run_time_sample,
                                      reset_clims=reset_clims)

                if run_time_reinit > 0.0: # if there is a reinitialization phase (RNAi applied, no cutting)
                    self.model.reinitialize(knockdown=rnai_s,
                                        run_time=run_time_reinit,
                                        run_time_step=run_time_step,
                                        run_time_sample=run_time_sample
                                       )
                # Run the simulation with RNAi intervention applied:
                self.model.simulate(knockdown=rnai_s,
                                run_time=run_time_sim,
                                run_time_step=run_time_step,
                                run_time_sample=run_time_sample,
                                reset_clims=reset_clims)

                # Save whole molecules master arrays to their respective data dictionaries:
                data_dict_inits[rnai_n] = self.model.molecules_time.copy()
                data_dict_sims[rnai_n] = self.model.molecules_sim_time.copy()

                self.model.process_markov(self.head_frags, self.tail_frags)
                data_dict_prob[rnai_n] = self.model.morph_probs.copy()

                if plot:
                    self.plot_single(rnai_n, 0, harness_type='simRNAi', plot_type=plot_type, axisoff = axisoff,
                                     output_type='sim', ref_data=self.ref_data[1], fsize=fsize, clims = clims)


                if animate:
                    self.ani_single(rnai_n, 0, harness_type='simRNAi', ani_type=ani_type, axisoff = axisoff,
                                    output_type='sim', ref_data=self.ref_data[1], fsize=fsize, clims = clims)

                if verbose is True:
                    print('----------------')

        self.outputs.append([data_dict_inits, data_dict_sims])
        self.heteromorphoses.append(data_dict_prob)

        if data_output:
            self.output_heteromorphs(save_dir=self.savedir_simRNAi)
            self.model.plot_frags(dir_save=self.savedir_simRNAi, fsize=fsize[0])

        if save_all:

            fsave = os.path.join(self.savedir_simRNAi, "Master.gz")
            self.save(fsave)


    def save(self, fname):

        if self.verbose:
            print("Saving harness results to file...")

        pickles.save(self, filename=fname, is_overwritable=True)

    def load(self, fname):

        if self.verbose:
            print("Loaded saved harness...")

        master = pickles.load(fname)

        return master

    def plot_all_output(self, loadpath, save_dir = 'Plots', plot_type=['Triplot'], output_type='sim',
                        autoscale = False, clims = None, cmaps = None, axisoff = False, fsize=[(6,8)]):

        load_fname = os.path.join(loadpath, "Master.gz")
        master = self.load(load_fname)

        dirname = os.path.join(loadpath, save_dir)

        if output_type == 'init':
            ref_data = master.ref_data[0]
        elif output_type == 'sim':
            ref_data = master.ref_data[1]

        if len(master.outputs):

            for plot_typei, fsizei in zip(plot_type, fsize):

                if output_type == 'init':

                    for ri, (inits_dict, sims_dict) in enumerate(master.outputs):

                        for init_i, tagi in zip(inits_dict.values(), master.datatags):

                            if self.verbose is True:
                                print('Plotting init of', tagi, 'for run ', ri, '...')

                            fni = plot_typei + '_' + tagi + '_init_' + str(ri)

                            master.model.molecules_time = init_i

                            if ri > 0:
                                master.write_plot_msg(ri)

                            if plot_typei == 'Triplot':
                                master.model.triplot(-1, plot_type='init', fname=fni, dirsave=dirname, fsize=fsizei,
                                                   cmaps=cmaps, clims=clims, autoscale=autoscale, axisoff = axisoff,
                                                   ref_data = ref_data, extra_text = master.plot_info_msg)

                            elif plot_typei == 'Biplot':
                                master.model.biplot(-1, plot_type='init', fname=fni, dirsave=dirname, fsize=fsizei,
                                                   cmaps=cmaps, clims=clims, autoscale=autoscale, axisoff = axisoff,
                                                  ref_data = ref_data, extra_text = master.plot_info_msg)

                            elif plot_typei == 'Markovplot':
                                master.model.markovplot(-1, plot_type='init', fname=fni, dirsave=dirname, fsize=fsizei,
                                                      cmaps=None, clims=None, autoscale=False, axisoff = axisoff,
                                                      ref_data=ref_data, extra_text=master.plot_info_msg)

                elif output_type == 'sim':

                    for ri, (inits_dict, sims_dict) in enumerate(master.outputs):

                        if ri > 0:
                            master.write_plot_msg(ri)

                        for sim_i, tagi in zip(sims_dict.values(), master.datatags):

                            if self.verbose is True:
                                print('Plotting sim of', tagi, 'for run ', ri, '...')

                            fns = plot_typei + '_' + tagi + '_sim_' + str(ri)
                            master.model.molecules_sim_time = sim_i

                            if plot_typei == 'Triplot':
                                master.model.triplot(-1, plot_type='sim', fname=fns, dirsave=dirname, fsize=fsizei,
                                                   cmaps=cmaps, clims=clims, autoscale=autoscale, axisoff = axisoff,
                                                   ref_data = ref_data, extra_text = master.plot_info_msg)

                            elif plot_typei == 'Biplot':
                                master.model.biplot(-1, plot_type='sim', fname=fns, dirsave=dirname, fsize=fsizei,
                                                  cmaps=cmaps, clims=clims, autoscale=autoscale, axisoff = axisoff,
                                                  ref_data = ref_data, extra_text = master.plot_info_msg)

                            elif plot_typei == 'Markovplot':
                                master.model.markovplot(-1, plot_type='sim', fname=fns, dirsave=dirname, fsize=fsizei,
                                                      cmaps=None, clims=None, autoscale=False, axisoff = axisoff,
                                                      ref_data=ref_data, extra_text=master.plot_info_msg)

                else:

                    print("Output type option must be 'init' or 'sim'.")

        else:

            print('No outputs to plot.')

    def ani_all_output(self, loadpath, save_dir = 'Animations', ani_type='Triplot', output_type='sim',
                       autoscale = False, cmaps = None, clims = None, axisoff = False, fsize=[(6,8)]):

        fsize = fsize[0]

        load_fname = os.path.join(loadpath, "Master.gz")
        master = self.load(load_fname)

        dirname = os.path.join(loadpath, save_dir)

        if output_type == 'init':
            ref_data = master.ref_data[0]
        elif output_type == 'sim':
            ref_data = master.ref_data[1]


        if len(master.outputs):

            if output_type == 'init':

                for ri, (inits_dict, sims_dict) in enumerate(master.outputs):

                    for init_i, tagi in zip(inits_dict.values(), master.datatags):

                        if self.verbose is True:
                            print('Animating init of', tagi, 'for run ', ri, '...')

                        dni = ani_type + '_' + tagi + '_init'+ str(ri)
                        plotdiri = os.path.join(dirname, dni)

                        master.model.molecules_time = init_i

                        if ri > 0:
                            master.write_plot_msg(ri)

                        if ani_type == 'Triplot':
                            self.model.animate_triplot(ani_type='init', dirsave=plotdiri, axisoff = axisoff,
                                               cmaps=cmaps, clims=clims, autoscale=autoscale, fsize=fsize,
                                                       ref_data = ref_data, extra_text = master.plot_info_msg)

                        elif ani_type == 'Biplot':
                            self.model.animate_biplot(ani_type='init', dirsave=plotdiri, axisoff = axisoff,
                                              cmaps=cmaps, clims=clims, autoscale=autoscale, fsize=fsize,
                                                      ref_data = ref_data, extra_text = master.plot_info_msg)

            elif output_type == 'sim':

                for ri, (inits_dict, sims_dict) in enumerate(master.outputs):

                    for sim_i, tagi in zip(sims_dict.values(), master.datatags):

                        if self.verbose is True:
                            print('Plotting sim of', tagi, 'for run ', ri, '...')

                        dns = ani_type + '_' + tagi + '_sim' + str(ri)
                        plotdirs =os.path.join(dirname, dns)

                        master.model.molecules_sim_time = sim_i

                        if ri > 0:
                            master.write_plot_msg(ri)

                        if ani_type == 'Triplot':
                            master.model.animate_triplot(ani_type='sim', dirsave=plotdirs, axisoff = axisoff,
                                               cmaps=cmaps, clims=clims, autoscale=autoscale, fsize=fsize,
                                                       ref_data = ref_data, extra_text = master.plot_info_msg)

                        elif ani_type == 'Biplot':
                            master.model.animate_biplot(ani_type='sim', dirsave=plotdirs, axisoff = axisoff,
                                              cmaps=cmaps, clims=clims, autoscale=autoscale, fsize=fsize,
                                                      ref_data = ref_data, extra_text = master.plot_info_msg)

            else:

                print("Output type option must be 'init' or 'sim'.")

        else:

            print('No outputs to animate.')

    def plot_single(self, tagi, ri, harness_type=None, plot_type=['Triplot'], output_type='sim',
                    ref_data = None, extra_text = None, axisoff = False, fsize = [(6,8)], clims = None):
        """

        :param tagi: datatag for the plot
        :param ri: itteration index of the model
        :param harness_type:  type of harness being simulated
        :param plot_type: plot a 'Triplot' or a 'Biplot'
        :param output_type:  plot an 'init' or a 'sim'
        :return: None
        """

        for plot_typei, fsizei in zip(plot_type, fsize):

            if harness_type is None:
                harness_type = ''
                plotdirmain = self.savepath

            else:
                plotdirmain = self.subfolders_dict[harness_type]

            if output_type == 'init':

                if self.verbose is True:
                    print('Plotting init of', tagi, 'for run ', ri, '...')

                fni = plot_typei + '_' + tagi + '_init_' + str(ri)

                if plot_typei == 'Triplot':
                    self.model.triplot(-1, plot_type='init', fname=fni, dirsave=plotdirmain, fsize=fsizei,
                                       cmaps=None, clims=clims, autoscale=False, axisoff = axisoff,
                                       ref_data = ref_data, extra_text = extra_text)

                elif plot_typei == 'Biplot':
                    self.model.biplot(-1, plot_type='init', fname=fni, dirsave=plotdirmain, fsize=fsizei,
                                       cmaps=None, clims=clims, autoscale=False, axisoff = axisoff,
                                      ref_data = ref_data, extra_text = extra_text)

                elif plot_typei == 'Hexplot':
                    self.model.hexplot(-1, plot_type='init', fname=fni, dirsave=plotdirmain, fsize=fsizei,
                                       cmaps=None, clims=clims, autoscale=False, axisoff = axisoff,
                                      ref_data = ref_data, extra_text = extra_text)

                elif plot_typei == 'Markovplot':
                    self.model.markovplot(-1, plot_type='init', fname=fni, dirsave=plotdirmain, fsize=fsizei,
                                       cmaps=None, clims=clims, autoscale=False, axisoff = axisoff,
                                       ref_data = ref_data, extra_text = extra_text)

            elif output_type == 'sim':

                if self.verbose is True:
                    print('Plotting sim of', tagi, 'for run ', ri, '...')

                fns = plot_typei + '_' + tagi + '_sim_' + str(ri)

                if plot_typei == 'Triplot':
                    self.model.triplot(-1, plot_type='sim', fname=fns, dirsave=plotdirmain, fsize=fsizei,
                                       cmaps=None, clims=clims, autoscale=False, axisoff = axisoff,
                                       ref_data = ref_data, extra_text = extra_text)

                elif plot_typei == 'Biplot':
                    self.model.biplot(-1, plot_type='sim', fname=fns, dirsave=plotdirmain, fsize=fsizei,
                                      cmaps=None, clims=clims, autoscale=False, axisoff = axisoff,
                                      ref_data = ref_data, extra_text = extra_text)

                elif plot_typei == 'Hexplot':
                    self.model.hexplot(-1, plot_type='sim', fname=fns, dirsave=plotdirmain, fsize=fsizei,
                                      cmaps=None, clims=clims, autoscale=False, axisoff = axisoff,
                                      ref_data = ref_data, extra_text = extra_text)

                elif plot_typei == 'Markovplot':
                    self.model.markovplot(-1, plot_type='sim', fname=fns, dirsave=plotdirmain, fsize=fsizei,
                                       cmaps=None, clims=clims, autoscale=False, axisoff = axisoff,
                                       ref_data = ref_data, extra_text = extra_text)

    def ani_single(self, tagi, ri, harness_type=None, ani_type='Triplot', output_type='sim', axisoff = True,
                   ref_data = None, extra_text = None, fsize=[(6,8)], clims = None):

        fsize = fsize[0]

        if harness_type is None:
            harness_type = ''
            plotdirmain = self.savepath

        else:
            plotdirmain = self.subfolders_dict[harness_type]

        if output_type == 'init':

            if self.verbose is True:
                print('Animating init of', tagi, 'for run ', ri, '...')

            dni = ani_type + '_' + tagi + '_init'+ str(ri)
            plotdiri = os.path.join(plotdirmain, dni)

            if ani_type == 'Triplot':
                self.model.animate_triplot(ani_type='init', dirsave=plotdiri, axisoff = axisoff,
                                   cmaps=None, clims=clims, autoscale=False, fsize=fsize,
                                           ref_data = ref_data, extra_text = extra_text)

            elif ani_type == 'Biplot':
                self.model.animate_biplot(ani_type='init', dirsave=plotdiri, axisoff = axisoff,
                                  cmaps=None, clims=clims, autoscale=False, fsize=fsize,
                                          ref_data = ref_data, extra_text = extra_text)

        elif output_type == 'sim':

            if self.verbose is True:
                print('Animating sim of', tagi, 'for run ', ri, '...')

            dns = ani_type + '_' + tagi + '_sim'+ str(ri)
            plotdirs =os.path.join(plotdirmain, dns)

            if ani_type == 'Triplot':
                self.model.animate_triplot(ani_type='sim', dirsave=plotdirs, axisoff = axisoff,
                                   cmaps=None, clims=clims, autoscale=False, fsize=fsize,
                                           ref_data = ref_data, extra_text = extra_text)

            elif ani_type == 'Biplot':
                self.model.animate_biplot(ani_type='sim', dirsave=plotdirs, axisoff = axisoff,
                                  cmaps=None, clims=clims, autoscale=False, fsize=fsize,
                                          ref_data = ref_data, extra_text = extra_text)

    def write_plot_msg(self, ii):

        if self.has_autoparams:

            # calculate percent changes to input variables in this run
            percent_delta_input = 100 * ((self.pm.params_M[ii, :] -
                                          self.pm.params_M[0, :]) / self.pm.params_M[0, :])

            index_delta = (percent_delta_input != 0.0).nonzero()[0]

            delta_i = np.round(percent_delta_input[index_delta], 1)
            name_i = self.pm.param_labels[index_delta]

            # add text to plot describing which parameters were changed in this run,
            # and by how much:
            param_changes_msg = ''
            if len(delta_i):

                for si, (ni, di) in enumerate(zip(name_i, delta_i)):
                    msg_i = 'Δ' + ni + '  ' + str(di) + '%\n'
                    param_changes_msg += msg_i

            self.plot_info_msg = param_changes_msg


    def work_all_output(self, save_dir = 'DataOutput', ii =-1, run_type = 'init',
                        substance = 'Erk', ref_data = None):

        tab_inputs = []
        tab_outputs = []

        master = self

        if len(master.outputs):

            if run_type == 'init':

                for ri, (inits_dict, sims_dict) in enumerate(master.outputs):

                    for init_i, tagi in zip(inits_dict.values(), master.datatags):

                        master.model.molecules_time = init_i

                        # if ri > 0:
                        out_diff = master.model.work_data(ti =ii, run_type = 'init',
                                                          substance = substance, ref_data = ref_data)

                        tab_outputs.append(out_diff * 1)

                        if self.has_autoparams:
                            # calculate percent changes to input variables in this run:
                            percent_delta_input = 100 * ((self.pm.params_M[ri, :] -
                                                          self.pm.params_M[0, :]) / self.pm.params_M[0, :])

                        else:
                            percent_delta_input = 0.0

                        tab_inputs.append(percent_delta_input * 1)

            elif run_type == 'sim':

                for ri, (inits_dict, sims_dict) in enumerate(master.outputs):

                    for sim_i, tagi in zip(sims_dict.values(), master.datatags):

                        master.model.molecules_sim_time = sim_i

                        # if ri > 0:
                        out_diff = master.model.work_data(ti=ii, run_type='sim',
                                                          substance=substance, ref_data=ref_data)

                        tab_outputs.append(out_diff * 1)

                        if self.has_autoparams:
                            # calculate percent changes to input variables in this run
                            percent_delta_input = 100 * ((self.pm.params_M[ri, :] -
                                                          self.pm.params_M[0, :]) / self.pm.params_M[0, :])

                        else:
                            percent_delta_input = 0.0

                        tab_inputs.append(percent_delta_input * 1)

            else:

                print("Output type option must be 'init' or 'sim'.")

        else:

            print('No outputs to work.')

        tab_inputs = np.asarray(tab_inputs)
        tab_outputs = np.asarray(tab_outputs)

        return tab_inputs, tab_outputs


    def output_delta_table(self, substance = ['Erk'], run_type = 'sim', save_dir = 'DataOutput'):

        for mol in substance:

            change_input, change_output = self.work_all_output(substance = mol,
                                                                         run_type = run_type,
                                                                         ref_data = self.ref_data)

            hdr = ''

            for plab in self.pm.param_labels:
                hdr += '% Δ' + plab + ','
            hdr += '% ΔOutput'

            writeM = np.column_stack((change_input[1:], change_output[1:]))

            fnme = mol + '_analysis.csv'

            fpath = os.path.join(save_dir, fnme)

            np.savetxt(fpath, writeM, delimiter=',', header=hdr)

    def output_summary_table(self, substance = ['Erk'], run_type = 'sim', save_dir = 'DataOutput'):

        for mol in substance:

            change_input, change_output = self.work_all_output(substance=mol,
                                                                         run_type = run_type,
                                                                         ref_data = self.ref_data)

            a, b = change_input[1:].shape

            if a == b:
                diag_vals = np.diag(change_input[1:])

                base_vals = np.array(list(self.paramo.values()))
                units = np.array(list(self.params_units.values()))

                hdr = 'Parameter, Base value, Units, %Δ Parameter, %Δ Output'

                writeM = np.column_stack((self.pm.param_labels, base_vals, units, diag_vals, change_output[1:]))

                fnme = mol + '_sensitivity_analysis.csv'

                fpath = os.path.join(save_dir, fnme)

                np.savetxt(fpath, writeM, fmt='%s', delimiter=',', header=hdr)

            head = ''
            for pi in self.pm.param_labels:
                head += pi + ','

            fpath2 = os.path.join(save_dir, 'param_values.csv')
            np.savetxt(fpath2, self.pm.params_M, delimiter=',', header=head)


    def output_heteromorphs(self, save_dir = 'DataOutput'):

        morph_col_tags = '2T,' + '0H,' + '1H,' + '0T,' + '2H,' + '00'

        for ri, hetmorphs_masterdict in enumerate(self.heteromorphoses):

            dir_path = os.path.join(save_dir, 'Run_' + str(ri))
            os.makedirs(dir_path, exist_ok=True)

            for tag_n, hmorphs_dict in hetmorphs_masterdict.items():

                morph_data = []

                for frag_n, hmorphs, in hmorphs_dict.items():
                    p2T = hmorphs['2T']
                    p0H = hmorphs['0H']
                    p1H = hmorphs['1H']
                    p0T = hmorphs['0T']
                    p2H = hmorphs['2H']
                    p00 = hmorphs['00']

                    row_data = [p2T, p0H, p1H, p0T, p2H, p00]

                    morph_data.append(row_data)

                morph_data = np.asarray(morph_data)

                fpath = os.path.join(dir_path, tag_n + '_heteromorphs.csv')

                np.savetxt(fpath, morph_data, delimiter=',', header=morph_col_tags)

    def output_heteromorphs_single(self, tagi, ri, save_dir = 'DataOutput'):

        morph_col_tags = '2T,' + '0H,' + '1H,' + '0T,' + '2H,' + '00'

        dir_path = os.path.join(save_dir, 'Run_' + str(ri))
        os.makedirs(dir_path, exist_ok=True)

        hmorphs_dict = self.model.morph_probs

        morph_data = []

        for frag_n, hmorphs, in hmorphs_dict.items():
            p2T = hmorphs['2T']
            p0H = hmorphs['0H']
            p1H = hmorphs['1H']
            p0T = hmorphs['0T']
            p2H = hmorphs['2H']
            p00 = hmorphs['00']

            row_data = [p2T, p0H, p1H, p0T, p2H, p00]

            morph_data.append(row_data)

        morph_data = np.asarray(morph_data)

        fpath = os.path.join(dir_path, tagi + '_heteromorphs.csv')

        np.savetxt(fpath, morph_data, delimiter=',', header=morph_col_tags)

    def output_clims(self, ri = 0, save_dir = 'DataOutput', run_type = 'init'):
        """
        Outputs min/max values used on plots for each substance.

        """

        # output clims for the run

        dir_path = os.path.join(save_dir, 'Run_' + str(ri))
        os.makedirs(dir_path, exist_ok=True)

        if run_type == 'init':
            fpath = os.path.join(dir_path, 'Clims_init.csv')

        elif run_type == 'reinit':
            fpath = os.path.join(dir_path, 'Clims_reinit.csv')

        elif run_type == 'sim':
            fpath = os.path.join(dir_path, 'Clims_sim.csv')

        MM = []
        for ctag, (cmin, cmax) in self.model.default_clims.items():
            MM.append([ctag, cmin, cmax])
        MM = np.asarray(MM)
        head = 'Substance, Min, Max'

        np.savetxt(fpath, MM, delimiter=',', fmt="%s", header=head)

    def view_fragments(self, fsize=(6,8)):

        # create a model using the specific parameters from the params manager for this run:
        self.model.model_init(self.config_fn, self.paramo, xscale=self.xscale,
                              verbose=self.verbose, new_mesh=self.new_mesh)

        if self.harness_type == '2D':

            self.model.cut_cells()

            self.model.plot_frags(dir_save=self.savepath, fsize=fsize)




