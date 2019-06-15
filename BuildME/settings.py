"""
Settings and constants required for the model to run.

Copyright: Niko Heeren, 2019
"""

import os

ep_version = '9.0.1'
basepath = os.path.abspath('.')
ep_path = os.path.abspath("./bin/EnergyPlus-9-0-1/")
ep_idd = os.path.abspath("./bin/EnergyPlus-9-0-1/Energy+.idd")
ep_exec_files = ["energyplus", "energyplus-%s" % ep_version, "Energy+.idd", "EPMacro", "ExpandObjects",
                 "libenergyplusapi.%s.dylib" % ep_version,  # required by energyplus
                 "libgfortran.3.dylib", "libquadmath.0.dylib",  # required by ExpandObjects
                 "PreProcess/GrndTempCalc/Basement", "PreProcess/GrndTempCalc/BasementGHT.idd",
                 "PostProcess/ReadVarsESO"
                 ]
archetypes = os.path.abspath("./data/archetype/")
tmp_path = os.path.abspath("./tmp/")
climate_files_path = os.path.abspath("./data/climate/meteonorm71/")

# The combinations
#   Example: USA.SFH_standard.RES0.
debug_combinations = {
    'USA':
        {'occupation': ['SFH', 'MFH'],
         'energy standard': ['standard', 'efficient', 'ZEB'],
         'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
         'climate_region': ['8'],
         'climate_scenario': ['2015']}}

combinations = \
    {
        'all':  # maybe not necessary
            {'occupation_types':
                 ['SFH_non-standard', 'SFH_standard', 'SFH_efficient', 'SFH_ZEB',
                  'MFH_non-standard', 'MFH_standard', 'MFH_efficient', 'MFH_ZEB',
                  'informal_non-standard'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_scenario':
                 ['2015',
                  '2030_A1B', '2030_A2', '2030_B1',
                  '2050_A1B', '2050_A2', '2050_B1']},
        'USA':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['1A', '2A', '2B', '3A', '3B-Coast', '3B', '3C',
                  '4A', '4B', '4C',
                  '5A', '5B', '6A', '6B', '7', '8'],
             'climate_scenario': ['2015']
             },
        'DE':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['Germany'],
             'climate_scenario': ['2015']
             },
        'CN':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['I', 'II', 'III', 'IV', 'V'],
             'climate_scenario': ['2015']
             },
        'JP':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['JP1', 'JP2', 'JP3', 'JP4', 'JP5', 'JP6', 'JP7', 'JP8'],
             'climate_scenario': ['2015']
             },
        'IT':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['Italy'],
             'climate_scenario': ['2015']
             },
        'FR':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['France'],
             'climate_scenario': ['2015']
             },
        'PL':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['Poland'],
             'climate_scenario': ['2015']
             },
        'CA':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['CA5A', 'CA6A', 'CA7'],
             'climate_scenario': ['2015']
             },
        'R32EU12-M':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['R32EU12-M'],
             'climate_scenario': ['2015']
             },
        'IN':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['IN1', 'IN2', 'IN3', 'IN4', 'IN5'],
             'climate_scenario': ['2015']
             },
        'ES':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['Spain'],
             'climate_scenario': ['2015']
             },
        'UK':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['UK'],
             'climate_scenario': ['2015']
             },
        'Oth-R32EU15':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['Oth-R32EU15'],
             'climate_scenario': ['2015']
             },
        'Oth-R32EU12-H':
            {'occupation': ['SFH', 'MFH'],
             'energy standard': ['non-standard', 'standard', 'efficient', 'ZEB'],
             'RES': ['RES0', 'RES2.1', 'RES2.2', 'RES2.1+RES2.2'],
             'climate_region':
                 ['Oth-R32EU12-H'],
             'climate_scenario': ['2015']
             }
    }

climate_stations = {
    'USA': {
        '1A': 'Miami_FL-hour.epw',
        '2A': 'Houston_Airp_TX-hour.epw',
        '2B': 'Phoenix_AZ-hour.epw',
        '3A': 'Atlanta_GA-hour.epw',
        '3B-Coast': 'Los_Angeles_CA-hour.epw',
        '3B': 'Las_Vegas_NV-hour.epw',
        '3C': 'San_Francisco_CA-hour.epw',
        '4A': 'Baltimore_MD-hour.epw',
        '4B': 'US-Albuquerque_NM-hour.epw',
        '4C': 'Seattle_Tacoma_WA-hour.epw',
        '5A': 'Chicago_IL-hour.epw',
        '5B': 'Boulder_CO-hour.epw',
        '6A': 'Minneapolis_Airp_MN-hour.epw',
        '6B': 'Helena_MT-hour.epw',
        '7': 'Duluth_Airp_MN-hour.epw',
        '8': 'Fairbanks_AK-hour.epw'},
    'DE': {'Germany': 'Frankfurt_am_Main_Airp_-hour.epw'},
    'CN': {'I': 'CN-Harbin.epw',
           'II': 'CN-Beijing.epw',
           'III': 'CN-Wuhan.epw',
           'IV': 'CN-Haikou.epw',
           'V': 'CN-Kunming.epw'},
    'JP': {
        'JP1': 'JP-Asahikawa-hour.epw',
        'JP2': 'JP-Sapporo-hour.epw',
        'JP3': 'JP-Morioka-hour.epw',
        'JP4': 'JP-Sendai-hour.epw',
        'JP5': 'JP-Tsukuba_JA-hour.epw',
        'JP6': 'JP-Osaka-hour.epw',
        'JP7': 'JP-Miyazaki-hour.epw',
        'JP8': 'JP-Naha-hour.epw'},
    'IT': {'Italy': 'Roma_Ciampino-hour.epw'},
    'FR': {'France': 'PARIS_FR-hour.epw'},
    'PL': {'Poland': 'PL-Warszawa-hour.epw'},
    'CA': {'CA5A': 'Chicago_IL-hour.epw',
           'CA6A': 'Minneapolis_Airp_MN-hour.epw',
           'CA7': 'Duluth_Airp_MN-hour.epw'},
    'R32EU12-M': {'R32EU12-M': 'PL-Warszawa-hour.epw'},
    'IN': {
        'IN1': 'IN-Jodhpur.epw',
        'IN2': 'IN-Santacruz_Bombay.epw',
        'IN3': 'IN-Bangalore.epw',
        'IN4': 'IN-Shimla.epw',
        'IN5': 'IN-Hyderabad.epw'},
    'ES': {'Spain': 'Madrid_Barajas-hour.epw'},
    'UK': {'UK': 'Aughton-hour.epw'},
    'Oth-R32EU15': {'Oth-R32EU15': 'PL-Warszawa-hour.epw'},
    'Oth-R32EU12-H': {'Oth-R32EU12-H': 'PL-Warszawa-hour.epw'}
}

odym_materials = {'Asphalt_shingle': 'other',
                  'Air_4_in_vert': 'other',
                  'Bldg_paper_felt': 'paper and cardboard',
                  'Std Wood 6inch': 'wood and wood products',
                  'Std Wood 10cm': 'wood and wood products',
                  'Lumber_2x4': 'wood and wood products',
                  'OSB_1/2in': 'wood and wood products',
                  'OSB_5/8in': 'wood and wood products',
                  'Stucco_1in': 'cement',
                  'sheathing_consol_layer': 'wood and wood products',
                  'Drywall_1/2in': 'wood and wood products',
                  'ceil_consol_layer-en-non-standard': 'other',
                  'ceil_consol_layer-en-standard': 'other',
                  'ceil_consol_layer-en-efficient': 'other',
                  'ceil_consol_layer-en-ZEB': 'other',
                  'door_const': 'wood and wood products',
                  'Glass-en-non-standard': 'other',
                  'Glass-en-standard': 'other',
                  'Glass-en-efficient': 'other',
                  'Glass-en-ZEB': 'other',
                  'Plywood_3/4in': 'wood and wood products',
                  'Carpet_n_pad': 'other',
                  'floor_consol_layer': 'wood and wood products',
                  'Concrete_20cm': 'concrete',
                  'Reinforcement_1perc_20cm': 'construction grade steel',
                  'Concrete_15cm': 'concrete',
                  'Reinforcement_1perc_15cm': 'construction grade steel',
                  'Concrete_12cm': 'concrete',
                  'Reinforcement_1perc_12cm': 'construction grade steel',
                  'wall_consol_layer-en-non-standard': 'other',
                  'wall_consol_layer-en-standard': 'other',
                  'wall_consol_layer-en-efficient': 'other',
                  'wall_consol_layer-en-ZEB': 'other'}

odym_regions = {'USA': 'R32USA',
                'CA': 'R32CAN',
                'CN': 'R32CHN',
                'R32EU12-M': 'R32EU12-M',
                'IN': 'R32IND',
                'JP': 'R32JPN',
                'FR': 'France',
                'DE': 'Germany',
                'IT': 'Italy',
                'PL': 'Poland',
                'ES': 'Spain',
                'UK': 'UK',
                'Oth-R32EU15': 'Oth-R32EU15',
                'Oth-R32EU12-H': 'Oth-R32EU12-H'}
