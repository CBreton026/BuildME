"""
Functions to manipulate the IDF files.

Copyright: Niko Heeren, 2019
"""
import collections
import logging
import pandas as pd
import numpy as np
from eppy.modeleditor import IDF
from collections import defaultdict
from BuildME import settings


class SurrogateElement:
    # TODO(cbreton026): Make SurrogateElement inherit from dict?
    # see: https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/
    """
    A surrogate class for windows and doors, because e.g. idf.idfobjects['Window'.upper()] does not contain an 'area'
    attribute. See also https://github.com/santoshphilip/eppy/issues/230.
    """

    def __init__(self, g):
        if type(g) == dict:
            self.area = g['area']
            self.Building_Surface_Name = g['Building_Surface_Name']
            self.Construction_Name = g['Construction_Name']
            self.key = g['key']
            self.Name = g['Name']
            self.fieldnames = [g.keys()]

        else:
            self.area = g.Length * g.Height
            self.Building_Surface_Name = g.Building_Surface_Name
            self.Construction_Name = g.Construction_Name
            self.key = g.key
            self.Name = g.Name
            self.fieldnames = g.fieldnames
            # See EnergyPlus idd file: the surface multiplier for fenestrationsurface:detailed,
            # door, glazed door, window is truncated to integer.
            self.Multiplier = int(g.Multiplier)

            # self.Outside_Boundary_Condition = g.Outside_Boundary_Condition
            # self.Zone_Name = g.Zone_Name
            # self.Surface_Type = g.Surface_Type

    def __repr__(self):
        return (f"{self.key}: {self.Name} \n"
                f"fieldnames: {self.fieldnames}")


""" FIXME(cbreton026: This is a 'soft' delete to see if this class is used anywhere)
class SurrogateMaterial:
    # TODO: DELETE?
    A surrogate class for materials, such as, because some material types (e.g. 'Material:NoMass') do not contain
    certain attributes that are later required (e.g. 'Density').

    def __init__(self, g):
        self.key = g.key
        self.Name = g.Name
        self.Density = None
"""


def extract_surfaces(idf, element_type, boundary, surface_type):
    """
    Fetches the elements from an IDF file and returns them in a list.
    :param idf: The IDF file
    :param element_type: The elements to be considered, e.g. ['BuildingSurface:Detailed', 'Window']
    :param boundary: "!- Outside Boundary Condition" as specified in the IDF, e.g. ['Outdoors']
    :param surface_type: "!- Surface Type" as specified in the IDF file, e.g. ['Wall']
    :return: List of eppy elements
    """
    surfaces = []
    for e in element_type:
        for s in idf.idfobjects[e.upper()]:
            if e not in ['Window', 'FenestrationSurface:Detailed']:
                if s.Outside_Boundary_Condition in boundary and s.Surface_Type in surface_type:
                    surfaces.append(s)
            else:
                if s.Outside_Boundary_Condition_Object in boundary and s.Surface_Type in surface_type:
                    surfaces.append(s)
    return surfaces


def extract_windows(idf):
    """
    Need a special function here, because eppy doesn't know the 'Window' object.
    If there are more use cases, this function can also be generlaized.
    :param idf:
    :return:
    """
    glazing = idf.idfobjects['Window'.upper()]
    windows = [SurrogateElement(g) for g in glazing]
    return windows


def extract_doors(idf):
    """
    Need a special function here, because eppy doesn't know the 'Door' object.
    If there are more use cases, this function can also be generlaized.
    :param idf:
    :return:
    """
    doors = idf.idfobjects['Door'.upper()]
    windows = [SurrogateElement(d) for d in doors]
    return windows


def flatten_surfaces(surfaces):
    """
    Just a simple function to flatten the surfaces dictionary created in get_surfaces() and return it as a list.
    :param surfaces: dictionary created by get_surfaces()
    :return: flat list of elements e.g. [BuildingSurface:Detailed,...]
    """
    flat = [[s for s in surfaces[sname]] for sname in surfaces]
    flat = [item for sublist in flat for item in sublist]
    return flat


def read_idf(in_file):
    # in_file = os.path.join(filepath, filename)
    idd = settings.ep_idd
    IDF.setiddname(idd)
    with open(in_file, 'r') as infile:
        idf = IDF(infile)
    return idf


def get_surfaces(idf, energy_standard, res_scenario):
    surfaces = {}

    # Define the surfaces to extract and their respective element_type, boundary and surface_type.
    target_surfaces = surfaces_to_extract()

    # Extract surfaces from the model
    for key, value in target_surfaces.items():
        surfaces.update({key: extract_surfaces(idf, value['element_type'], value['boundary'], value['surface_type'])})

    # Add some missed surfaces manually
    # FIXME(cbreton026): Could/should this be in extract surfaces directly?
    surfaces['door'].extend(extract_doors(idf))
    surfaces['window'].extend(extract_windows(idf))

    # Find out if there are surface multipliers applied to specific surfaces
    # in the model, and apply them. # TODO(cbreton026): change the variable name
    surfaces_with_s_multipliers = apply_surface_multipliers(surfaces)
    if surfaces_with_s_multipliers:
        logging.warning("There are surface multipliers in model %s: %s" % (str(idf.idfname), surfaces_with_s_multipliers))

        for key in surfaces_with_s_multipliers:
            if key in surfaces:
                logging.warning("adding multiplied surfaces for %s" % key)
                surfaces[key].extend(surfaces_with_s_multipliers[key])
            else:
                logging.warning("Cannot add the multiplied surfaces for %s " % key)

    # Find out if there zone multipliers applied to specific zones in the model.
    zone_multipliers = get_zone_multipliers(idf)
    if zone_multipliers:
        logging.warning("There are zone multipliers in model %s: %s" % (str(idf.idfname), zone_multipliers))

        # Apply zone multipliers to the surfaces, if there are any
        multiplied_surfaces_to_add = apply_zone_multipliers(zone_multipliers, surfaces)

        for key in multiplied_surfaces_to_add:
            if key in surfaces:
                logging.warning("adding multiplied surfaces for %s" % key)
                surfaces[key].extend(multiplied_surfaces_to_add[key])
            else:
                logging.warning("Cannot add the multiplied surfaces for %s " % key)

    # Check for missed surfaces among relevant surfaces in the idf file.
    # 'FenestrationSurface:Detailed' is included as it is required for RT, and
    # including it does not affect calculations for other models.
    surfaces_to_count = ['Window', 'FenestrationSurface:Detailed',
                         'BuildingSurface:Detailed', 'Door']
    missed_surfaces = check_missed_surfaces(idf, surfaces_to_count, surfaces)
    if missed_surfaces:
        logging.warning("Following elements are not accounted for: %s" % missed_surfaces)

    # Add surrogate elements to the model
    # TODO(cbreton026): can these new surrogate elements be affected by multipliers?
    surrogate_elems = add_surrogate_elem(idf, surfaces, energy_standard, res_scenario)

    if surrogate_elems:
        logging.warning("Adding surrogate elements to the model.")

        for key in surrogate_elems:
            if key in surfaces:
                logging.warning("adding surrogate elements to surfaces['%s']" % key)
                surfaces[key].extend(surrogate_elems[key])
            else:
                logging.warning("adding a new surface and key for '%s'" % key)
                surfaces[key] = surrogate_elems[key]

    return surfaces


def create_surrogate_int_walls(floor_area, construction, linear_m=0.4, room_h=2.8):
    """
    Since IDF files sometimes do not contain internal walls, this function will create surrogate internal walls.
    Based on Kellenberger et al. 2012, 0.4 m per 1.0 m2 floor area is assumed. Assuming a room height of 2.8 m,
     this corresponds to  1.12 m2 per 1.0 m2 floor area.
    :return: List of one surface which can be added to the surfaces variable in get_surfaces().
    """
    int_wall = {
        'key': 'DummyBuildingSurface',
        'Name': 'surrogate_int_wall',
        'Building_Surface_Name': None,
        'Construction_Name': construction,
        'area': linear_m * floor_area * room_h
    }
    return [SurrogateElement(int_wall)]


def create_surrogate_slab(floor_area, construction):
    slab = {
        'key': 'DummyBuildingSurface',
        'Name': 'surrogate_slab',
        'Building_Surface_Name': None,
        'Construction_Name': construction,
        'area': floor_area
    }
    return [SurrogateElement(slab)]


def create_surrogate_basement(floor_area, construction, room_h=2.8):
    basem = {
        'key': 'DummyBuildingSurface',
        'Name': 'surrogate_basement',
        'Building_Surface_Name': None,
        'Construction_Name': construction,
        # assuming a square floor layout
        'area': floor_area ** 0.5 * 4 * room_h
    }
    return [SurrogateElement(basem)]


def create_surrogate_shear_wall(floor_area, construction):
    """
    The RT archetype need shear/core walls for lateral load resistance.
    Based on Taranath: Reinforced Concrete Design of Tall Buildings p. 144: 0.08 m per 1.0 m2 floor area is assumed.
    Assuming room height of 3 m, this yields 0.24 m2 per m2 floor area.

    :return: List of one surface which can be added to the surfaces variable in get_surfaces().
    """
    shear_walls = {
        'key': 'DummyBuildingSurface',
        'Name': 'surrogate_shear_wall',
        'Building_Surface_Name': None,
        'Construction_Name': construction,
        'area': 0.24 * floor_area
    }
    return [SurrogateElement(shear_walls)]


def calc_surface_areas(surfaces, floor_area=['int_floor', 'ext_floor']):
    """
    Sums the surfaces as created by get_surfaces() and returns a corresponding dict.
    :param floor_area:
    :param surfaces:
    :return:
    """
    areas = {}
    for element in surfaces:
        areas[element] = sum(e.area for e in surfaces[element])
    areas['ext_wall_area_net'] = areas['ext_wall'] - areas['window']
    areas['floor_area_wo_basement'] = sum([areas[s] for s in areas if s in floor_area])
    areas['footprint_area'] = areas['ext_floor']
    return areas


def calc_envelope(areas):
    """
    Calculates the total envelope surface area in the surfaces variable created by get_surfaces().
    :param areas:
    :return: Dictionary of surface area with and without basement
    """
    envelope = {
        'envelope_w_basement': sum(areas[s] for s in ['ext_wall', 'roof', 'ext_floor']),
        'envelope_wo_basement': areas['envelope_w_basement'] + areas['basement_ext_wall']}
    return envelope


def read_materials(idf):
    materials = []
    for mtype in ['Material', 'Material:NoMass', 'Material:AirGap',
                  'WindowMaterial:SimpleGlazingSystem', 'WindowMaterial:Blind',
                  'WindowMaterial:Glazing']:
        materials = materials + [i for i in idf.idfobjects[mtype.upper()]]
    find_duplicates(materials)
    # TODO: Will need to think about windows...
    return materials


def load_material_data():
    filedata = pd.read_excel('data/material.xlsx', sheet_name='properties', index_col='ep_name')
    return filedata


def find_duplicates(idf_object, crash=True):
    """
    Checks if duplicate entries in an IDF object exist
    :param crash:
    :param idf_object:
    :return: None
    """
    object_names = [io.Name for io in idf_object]
    duplicates = [item for item, count in collections.Counter(object_names).items() if count > 1]
    if crash:
        assert len(duplicates) == 0, "Duplicate entries for IDF object: '%s'" % duplicates
    else:
        return duplicates


def make_materials_dict(materials):
    """
    Takes the eppy materials objects and places them into a dictionary with the .Name attribute as the key,
    e.g. {material.Name: material}
    :param materials: list of eppy idf objects as created by read_materials()
    :return: dictionary with the .Name attribute as the key {material.Name: material}
    """
    # Making sure there are no duplicate Material entries in the IDF file
    materials_dict = {m.Name: m for m in materials}
    return materials_dict


def make_mat_density_dict(materials_dict, fallback_mat):
    """
    Creates a dictionary of material densities by material.
    TODO: Not sure if this can be derived from the IDF file only.
    :return:
    """
    densities = {}
    oopsies = []
    for mat in materials_dict:
        # Some materials, such as Material:No Mass have no density attribute
        if hasattr(materials_dict[mat], 'Density'):
            densities[mat] = materials_dict[mat].Density
        elif mat in fallback_mat.index:
            densities[mat] = fallback_mat.loc[mat, 'density']
        else:
            # print(mat, materials_dict[mat].key)
            oopsies.append(mat)
    if len(oopsies) != 0:
        raise AssertionError("Following materials have no density defined in idf Constructions nor in "
                             "data/materials.xlsx: %s"
                             % oopsies)
    return densities


def read_constructions(idf):
    """
    Gets the "Construction" elements from the idf files.
    :param idf:
    :return:
    """
    # TODO(cbreton026): sets are unordered, but have no duplicates. They might
    # be useful to replace find_duplicates?
    constructions = idf.idfobjects['Construction'.upper()]
    find_duplicates(constructions)
    return constructions


def extract_layers(construction):
    res = {}
    layers = ['Outside_Layer'] + ['Layer_' + str(i + 2) for i in range(9)]
    for layer in layers:
        if getattr(construction, layer) == '':
            break  # first empty value found
        res[layer] = getattr(construction, layer)
    return res


def get_fenestration_objects_from_surface(idf, surface_obj):
    """
    Finds all fenestration objects assigned to a given surface
    :param idf: The .idf file
    :param surface_obj: Surface object (BuildingSurface:Detailed)
    :return: list of fenestration objects
    """
    surface = surface_obj.Name
    fenestration = []
    for item in ['Window', 'Door', 'FenestrationSurface:Detailed']:
        new = [obj for obj in idf.idfobjects[item] if obj.Building_Surface_Name == surface]
        fenestration.extend(new)
    return fenestration


def surfaces_to_extract():
    # TODO(cbreton026): Eventually move this to a 'clean' version of settings.py
    surfaces = {
        'ext_wall':
            {'element_type': ['BuildingSurface:Detailed'],
             'boundary': ['Outdoors'],
             'surface_type': ['Wall']
             },
        'int_wall':
            {'element_type': ['BuildingSurface:Detailed'],
             'boundary': ['Surface', 'Zone'],
             'surface_type': ['Wall']
             },
        'basement_ext_wall':
            {'element_type': ['BuildingSurface:Detailed'],
             'boundary': ['GroundBasementPreprocessorAverageWall', 'GroundFCfactorMethod'],
             'surface_type': ['Wall']
             },
        'ext_floor':
            {'element_type': ['BuildingSurface:Detailed'],
             'boundary': ['Ground', 'GroundSlabPreprocessorAverage',
                          'GroundFCfactorMethod', 'Outdoors'],
             'surface_type': ['Floor']
             },
        'int_floor':
            {'element_type': ['BuildingSurface:Detailed'],
             'boundary': ['Adiabatic', 'Surface'],  # FIXME(@cbreton026): Remove 'Adiabatic'? See @shnkn #47
             'surface_type': ['Floor']
             },
        'basement_int_floor':
            {'element_type': ['BuildingSurface:Detailed'],
             'boundary': ['Zone'],
             'surface_type': ['Floor']
             },
        'int_ceiling':
            {'element_type': ['BuildingSurface:Detailed'],
             'boundary': ['Surface', 'Adiabatic'],
             'surface_type': ['Ceiling']
             },
        'ceiling_roof':
            {'element_type': ['BuildingSurface:Detailed'],
             'boundary': ['Zone'],
             'surface_type': ['Ceiling']},
        'roof':
            {'element_type': ['BuildingSurface:Detailed'],
             'boundary': ['Outdoors'],
             'surface_type': ['Roof']
             },
        'door':
            {'element_type': ['FenestrationSurface:Detailed'],
             'boundary': [''],
             'surface_type': ['Door']
             },
        'window':
            {'element_type': ['FenestrationSurface:Detailed'],
             'boundary': [''],
             'surface_type': ['Window', 'GlassDoor']
             }
    }
    return surfaces


def check_missed_surfaces(idf, surfaces_to_count, extracted_surfaces):
    """Checks if get_surfaces missed relevant surfaces in the idf.

    Args:
        idf (file):
            The idf file to scan.
        surfaces_to_count (list):
            A list of relevant surfaces to count in the
            idf, e.g. ['Window', 'BuildingSurface:Detailed', 'Door'].
        extracted_surfaces (dict):
            A dictionnary containing the extracted surfaces from get_surfaces().

    Returns:
        missed_surfaces (list):
            A list of the relevant surfaces that are present
            in the idf file, but absent from the extracted_surfaces.
    """
    missed_surfaces = []

    # Get total number of relevant surfaces in the idf.
    total_no_surfaces = [[surface for surface in idf.idfobjects[surface_name.upper()]]
                         for surface_name in surfaces_to_count]
    total_no_surfaces = [item for sublist in total_no_surfaces for item in sublist]

    # Identify any missed relevant surfaces in extracted_surfaces.
    missed_surfaces = [surface.Name for surface in total_no_surfaces
                       if surface.Name not in [extracted_surface.Name for extracted_surface
                                               in flatten_surfaces(extracted_surfaces)]]
    return missed_surfaces


def apply_surface_multipliers(surfaces):
    """Returns missing surfaces for all surface multipliers greater than 1 in the model.

    Args:
        surfaces (dict):
            A dictionary of surfaces in the idf file, where the key is the
            surface type, and the value is a list of surface objects. For example:
                {'ext_wall': [EpBunch, EpBunch],
                 'slab: [SurrogateElement],
                 'int_wall: [EpBunch, EpBunch]'}

    Returns:
        multiplied_surfaces_to_add (dict):
            a dict of multiplied surfaces that must be added to 'surfaces'.
    """
    multiplied_surfaces_to_add = {}

    for surface_key in surfaces:
        if not surfaces[surface_key]:
            logging.warning("surface %s is empty - skipping it" % surface_key)
            continue

        for surface in surfaces[surface_key]:
            if 'Multiplier' not in surface.fieldnames:
                continue
            elif int(surface.Multiplier) == 1:
                continue
            else:
                logging.warning(f"'{surface.Name}' has a surface multiplier of {surface.Multiplier}")
                multiplied_surfaces_to_add[surface_key].extend(np.repeat(surface, surface.Multiplier - 1).tolist())

    return multiplied_surfaces_to_add


def get_zone_multipliers(idf):
    """Checks for any zone multipliers greater than 1 in the model.

    Assesses if there are zone multipliers, and returns a dict containing
    all zones with a multiplier greater than one. For example:
        {'zone_name': zone_multiplier}
        {'M Corridor ZN Thermal Zone': 14,
         'M N1 Apartment ZN Thermal Zone': 14,
         'M N2 Apartment ZN Thermal Zone': 14}
    Args:
        idf (file):
            The idf file to scan.

    Raises:
        an: DESCRIPTION.

    Returns:
        zone_multipliers (dict): A dict of multipliers > 1.
    """
    zone_multipliers = {}

    zones = idf.idfobjects['ZONE']
    if not zones:
        # TODO(cbreton026): It may be better to raise an error if there are no zones?
        logging.warning(f"Empty list: BuildME recognizes no zones in the idf for this IDF: {idf.idfname}.")

    for zone in zones:
        if zone.Multiplier != '' and zone.Multiplier != 1:
            zone_multipliers[zone.Name] = zone.Multiplier

    return zone_multipliers


def apply_zone_multipliers(zone_multipliers, surfaces):
    """Returns missing surfaces for all zone multipliers greater than 1 in the model.


    Args:
        zone_multipliers (dict):
            A dictionary containing zone names and zone multipliers. See
            get_zone_multipliers().

        surfaces (dict):
            A dictionary of surfaces in the idf file, where the key is the
            surface type, and the value is a list of surface objects. For example:
                {'ext_wall': [EpBunch, EpBunch],
                 'slab: [SurrogateElement],
                 'int_wall: [EpBunch, EpBunch]'}

    Returns:
        multiplied_surfaces_to_add (dict):
            a dict of multiplied surfaces that must be added to 'surfaces'.

    """
    multiplied_surfaces_to_add = {}

    for surface_key in surfaces:
        multiplied_surface = []
        if not surfaces[surface_key]:
            logging.warning("surface %s is empty - skipping it" % surface_key)
            continue

        for surface in surfaces[surface_key]:
            # Get each surface's zone name.
            try:
                if 'Building_Surface_Name' in surface.fieldnames:
                    # FIXME(cbreton026): this works, but only for windows in 'ext_wall'.
                    # Find a more general approach, e.g. by looking in BuildingSurface:Detailed earlier.
                    zone_name = [item.Zone_Name for item in surfaces['ext_wall'] if item.Name == surface.Building_Surface_Name][0]
                    # [item.Zone_Name for item in idf.idfobjects['BuildingSurface:Detailed'] if item.Name == surface.Building_Surface_Name] more general - do earlier?
                    # [item.Zone_Name for item in idf_RT.idfobjects['BuildingSurface:Detailed'] if item.Name == 't WWall SWA']

                elif 'Zone_Name' in surface.fieldnames:
                    zone_name = surface.Zone_Name
                else:
                    # TODO(cbreton026): Possibly rework into a try/except or getattr()?
                    # https://stackoverflow.com/questions/610883/how-to-know-if-an-object-has-an-attribute-in-python
                    logging.warning("Unrecognized zone name; Zone_Name and Building_Surface_Name absent from fieldnames")  # TODO: Possible use an assert
            except AttributeError as err:
                logging.warning(f"The {err}. '{surface.Name}' is a {type(surface)}"
                                "and is not supported by apply_zone_multipliers()")

            # print(zone_name, surface.Name)
            if zone_name in zone_multipliers:
                logging.debug(f"multiplying surfaces for {zone_name}")
                multiplied_surface.extend(np.repeat(surface, zone_multipliers[zone_name] - 1).tolist())

        # surfaces_including_multipliers[surface_key].extend(multiplied_surface)
        multiplied_surfaces_to_add[surface_key] = multiplied_surface

    return multiplied_surfaces_to_add


def add_surrogate_elem(idf, surfaces, energy_standard, res_scenario):
    surrogate_elems = defaultdict(list)

    surface_areas = calc_surface_areas(surfaces)
    # TODO(cbreton026): gettint this I/O outside the loop would improve performance
    constr_dict = {m.Name: m for m in read_constructions(idf)}

    surrogate_list = [''.join(['attic-ceiling-', energy_standard]),
                      ''.join(['Surrogate_slab-', res_scenario]),
                      ''.join(['Concrete_floor_slab-', res_scenario]),
                      ''.join(['Shear_wall-', res_scenario])
                      ]

    # Check is the IDF is RT - it doesn't fit with the general form (yet - # FIXME(cbreton026))
    is_RT = 'RT.idf' in idf.idfname

    for elem in surrogate_list:
        if elem not in constr_dict.keys():
            continue

        # TODO(cbreton026): In python 3.10 this could be replaced by a match-case
        if 'attic-ceiling-' in elem:
            logging.warning(f"Adding {elem} to interior walls...")
            surrogate_elems['int_wall'].extend(
                create_surrogate_int_walls(surface_areas['floor_area_wo_basement'], elem)
            )

        elif 'Surrogate_slab-' in elem:
            if is_RT is True:
                # TODO(cbreton026): get_surface_with_zone_multipliers() adds two
                # slabs (slab, slab1) and basements (basement, basement1) to the model,
                # but this seems abitrary / specific to the model. This should be
                # generalized, but I'm not sure how to do it (yet).
                # Possibly use a dict? A multiplier?
                logging.warning(f"Adding {elem} to basement...")
                surrogate_elems['slab'].extend(
                    create_surrogate_slab(surface_areas['footprint_area'], elem))
                surrogate_elems['slab1'].extend(
                    create_surrogate_slab(surface_areas['footprint_area'], elem))
                surrogate_elems['basement'].extend(
                    create_surrogate_basement(surface_areas['footprint_area'], elem))
                surrogate_elems['basement1'].extend(
                    create_surrogate_basement(surface_areas['footprint_area'], elem))
            else:
                logging.warning(f"Adding {elem} to basement...")
                surrogate_elems['slab'].extend(
                    create_surrogate_slab(surface_areas['footprint_area'], elem)
                )
                surrogate_elems['basement'].extend(
                    create_surrogate_basement(surface_areas['footprint_area'], elem)
                )

        elif 'Concrete_floor_slab-' in elem:
            if is_RT is True:
                # TODO(cbreton026): get_surface_with_zone_multipliers() adds two
                # slabs (slab, slab1) and basements (basement, basement1) to the model,
                # but this seems abitrary / specific to the model. This should be
                # generalized, but I'm not sure how to do it (yet).
                # Possibly use a dict? A multiplier? Where would it be added?
                logging.warning(f"Adding {elem} to basement...")
                surrogate_elems['int_floor_second_floor'].extend(
                    create_surrogate_slab(surface_areas['footprint_area'], elem))
            else:
                # FIXME(cbreton026): would it still be added to int_floor_second_floor?
                logging.warning(f"Adding {elem} to basement...")
                surrogate_elems['int_floor_second_floor'].extend(
                    create_surrogate_slab(surface_areas['footprint_area'], elem))

        elif 'Shear_wall-' in elem:
            logging.warning(f"Adding {elem} to shear walls...")
            surrogate_elems['shear_wall'].extend(
                create_surrogate_shear_wall(surface_areas['floor_area_wo_basement'], elem)
            )

        else:
            logging.warning(f"'{elem}' is not recognized by BuildME."
                            "It was not added to the model.")

    return surrogate_elems
